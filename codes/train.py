import os
import logging
import argparse
import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from collections import defaultdict
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from custom_datasets.chip_dataset import ChipDataset
from custom_transform import BinaryMask16, EqualizeHist16
from network.attention_unet import AttentionUNet

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # conf = OmegaConf.load(cfg)
    # print(conf.pretty())

    # device 설정
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(
        "GPU_number : ",
        torch.cuda.current_device(),
        "\tGPU name:",
        torch.cuda.get_device_name(torch.cuda.current_device()),
    )

    # tensorboard 설정
    writer = SummaryWriter(os.path.join("../runs"))

    transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
            ],
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    }

    datasets = {
        "train": ChipDataset(
            data_dir_list=cfg.dataset.train_data_folder_list,
            transform=transforms["train"],
            **cfg.dataset,
        ),
        "test": ChipDataset(
            data_dir_list=cfg.dataset.test_data_folder_list,
            transform=transforms["test"],
            **cfg.dataset,
        ),
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.task.batch_size,
            shuffle=True,
            num_workers=cfg.task.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=cfg.task.batch_size,
            shuffle=False,
            num_workers=cfg.task.num_workers,
        ),
    }

    model = AttentionUNet(in_channels=1, out_channels=1, channels=cfg.model.channels)
    critic = nn.MSELoss()
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    epochs_pbar = tqdm(range(1, cfg.task.epochs), desc="epochs", position=0)
    for epoch in epochs_pbar:
        if epoch % cfg.task.test_interval:
            phases = ["train", "test"]
        else:
            phases = ["train"]

        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            num_epoch_samples = 0

            iter_pbar = tqdm(
                dataloaders["phase"], desc="iters", position=1, leave=False
            )
            for inputs, heatmaps in iter_pbar:
                inputs = inputs.to(device)
                heatmaps = heatmaps.to(device)
                num_iter_samples = inputs.size(0)
                num_epoch_samples += num_iter_samples

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    loss = critic(outputs, heatmaps)
                    metrics["loss"] += loss.item() * num_iter_samples

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

            epoch_loss = metrics["loss"] / num_epoch_samples
            # Loss를 tensorboard에 기록
            writer.add_scalar(f"{phase} loss", epoch_loss, global_step=epoch)

        scheduler.step()

        epochs_pbar.set_postfix(loss=epoch_loss)

    epochs_pbar.close()


if __name__ == "__main__":
    main()
