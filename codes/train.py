import os
import logging
import hydra
import torch
import torch.nn as nn
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

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
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

    # tensorboard 설정
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"runs/{current_time}")

    # profiler 설정
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"runs/{current_time}_prof"
        ),
        record_shapes=True,
        with_stack=True,
    )

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(degrees=30),
            ],
        ),
        "test": transforms.Compose([]),
    }

    datasets = {
        "train": ChipDataset(
            data_dir_list=cfg.train.train_data_folder_list,
            transform=data_transforms["train"],
            **cfg.dataset,
        ),
        "test": ChipDataset(
            data_dir_list=cfg.train.test_data_folder_list,
            transform=data_transforms["test"],
            **cfg.dataset,
        ),
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
        ),
    }

    model = AttentionUNet(in_channels=1, out_channels=1, channels=cfg.model.channels)
    model = model.to(device)
    critic = nn.MSELoss()
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    epoch_loss = 0.0

    # profiling 시작
    prof.start()

    epochs_pbar = tqdm(range(1, cfg.train.epochs), desc="epochs", position=0)
    for epoch in epochs_pbar:
        if epoch % cfg.train.test_interval:
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

            iter_pbar = tqdm(dataloaders[phase], desc="iters", position=1, leave=False)
            for inputs, heatmaps in iter_pbar:
                inputs = inputs.to(device)
                heatmaps = heatmaps.to(device)
                num_iter_samples = inputs.size(0)
                num_epoch_samples += num_iter_samples

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    loss = critic(outputs, heatmaps)
                    metrics["loss"] += 1000 * loss.item() * num_iter_samples

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

            epoch_loss = metrics["loss"] / num_epoch_samples

            # Loss를 tensorboard에 기록
            writer.add_scalar(f"Loss/{phase}", epoch_loss, global_step=epoch)

            prof.step()

        # learning rate를 tensorboard에 기록
        writer.add_scalar(f"lr", scheduler.get_last_lr()[0], global_step=epoch)
        scheduler.step()

        epochs_pbar.set_postfix(loss=epoch_loss)

    epochs_pbar.close()

    # profiling 종료
    prof.stop()

    # 모델 저장
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "optim": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    model_save_path = (
        cfg.train.model_save_path + f"_epoch_{epoch}_loss_{epoch_loss:.4f}.pth"
    )
    torch.save(state, model_save_path)


if __name__ == "__main__":
    train()
