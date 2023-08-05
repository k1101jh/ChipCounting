import os
import logging
import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from datetime import datetime
from hydra.utils import instantiate
from collections import defaultdict
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from custom_datasets.chip_dataset import ChipDataset
from network.attention_unet import AttentionUNet
from utils import plot_images

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def inference(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # device 설정
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(device)
    print(
        "GPU_number : ",
        torch.cuda.current_device(),
        "\tGPU name:",
        torch.cuda.get_device_name(torch.cuda.current_device()),
    )

    data_transform = transforms.Compose([])

    test_dataset = ChipDataset(
        data_dir_list=cfg.inference.inference_data_folder_list,
        transform=data_transform,
        **cfg.dataset,
    )

    dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.inference.batch_size,
        shuffle=False,
        num_workers=cfg.inference.num_workers,
    )

    model = AttentionUNet(in_channels=1, out_channels=1, channels=cfg.model.channels)
    model = model.to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(cfg.inference.saved_model_dir, cfg.inference.saved_model_file),
            map_location=device,
        )["model"]
    )
    model.eval()

    logging.info("Inference 시작")
    for inputs, heatmaps in tqdm(dataloader, desc="iters", leave=True):
        inputs = inputs.to(device)
        heatmaps = heatmaps.to(device)
        num_iter_samples = inputs.size(0)

        with torch.no_grad():
            outputs = model(inputs)

        # 이미지 시각화
        tensors_to_visualize = [inputs, heatmaps, outputs]
        labels = ["original image", "heatmap image", "output image"]

        for i, (tensor, label) in enumerate(zip(tensors_to_visualize, labels)):
            images = tensor / tensor.max()
            images = images.clamp(0, 1)
            images = torchvision.utils.make_grid(images).cpu()
            images = images.numpy().transpose((1, 2, 0))
            plt.subplot(len(labels), 1, i + 1)
            plt.imshow(images)
            plt.ylabel(label)
        plt.show()


if __name__ == "__main__":
    inference()
