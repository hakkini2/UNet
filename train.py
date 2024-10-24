from pathlib import Path

import fire
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from dataset.unet_dataset import get_train_transforms, get_val_transforms


def main(**kwargs):
    from_cli = OmegaConf.create(kwargs)
    base_conf = OmegaConf.load("./configs/_base_config.yaml")
    cfg = OmegaConf.merge(base_conf, from_cli)

    set_determinism(seed=cfg.seed)

    organ = cfg.organ.split("_")[1].lower()
    print(f"Training for {organ}")

    # Create dataset and dataloader
    image_files_train = sorted(
        [
            Path(f).as_posix()
            for f in Path(cfg.data_2d_path, "train_2d_images").iterdir()
            if f.is_file() and f.stem.startswith(organ)
        ]
    )
    label_files_train = sorted(
        [
            Path(f).as_posix()
            for f in Path(cfg.data_2d_path, "train_2d_masks").iterdir()
            if f.is_file() and f.stem.startswith(organ)
        ]
    )

    image_files_val = sorted(
        [
            Path(f).as_posix()
            for f in Path(cfg.data_2d_path, "val_2d_images").iterdir()
            if f.is_file() and f.stem.startswith(organ)
        ]
    )
    label_files_val = sorted(
        [
            Path(f).as_posix()
            for f in Path(cfg.data_2d_path, "val_2d_masks").iterdir()
            if f.is_file() and f.stem.startswith(organ)
        ]
    )

    # Create a list of dictionaries for the dataset
    data_dicts_train = [{"image": image, "label": label} for image, label in zip(image_files_train, label_files_train)]
    data_dicts_val = [{"image": image, "label": label} for image, label in zip(image_files_val, label_files_val)]

    # Train/val data transforms
    train_transforms = get_train_transforms(cfg.roi_size)
    val_transforms = get_val_transforms()

    train_ds = CacheDataset(
        data=data_dicts_train, transform=train_transforms, cache_rate=1.0, num_workers=cfg.num_workers
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.num_workers)
    val_ds = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=1.0, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.val_bs, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model definition
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
    ).to(device)

    loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=cfg.step_lr_size, gamma=cfg.gamma_lr)
    monai_dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Training loop
    max_epochs = cfg.n_epochs
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_train_values = []
    epoch_loss_val_values = []
    metric_values = []

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for _, batch_data in enumerate(train_loader):
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            if organ in ["liver", "hepaticvessel", "pancreas"]:
                labels[labels == 2] = 1
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= step
        epoch_loss_train_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            epoch_loss_val = 0
            step = 0
            monai_dice_metric.reset()
            for val_data in val_loader:
                step += 1
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device).long()

                if organ in ["liver", "hepaticvessel", "pancreas"]:
                    val_labels[val_labels == 2] = 1

                val_outputs = sliding_window_inference(val_inputs, (cfg.roi_size, cfg.roi_size), 4, model)
                val_outputs_loss = torch.stack([i for i in decollate_batch(val_outputs)], dim=0).to(device)
                loss = loss_function(val_outputs_loss, val_labels)
                epoch_loss_val += loss.item()

                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                monai_dice_metric(y_pred=val_outputs, y=val_labels)

            epoch_loss_val /= step
            epoch_loss_val_values.append(epoch_loss_val)
            metric = monai_dice_metric.aggregate().item()
            monai_dice_metric.reset()
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), f"{organ}_best_metric_model.pth")
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice (val): {metric:.4f} best mean dice (val): {best_metric:.4f} at epoch: {best_metric_epoch}"
            )
            print(f"Validation loss: {epoch_loss_val:.4f}")
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


if __name__ == "__main__":
    fire.Fire(main)
