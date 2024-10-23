from pathlib import Path

import fire
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from dataset.unet_dataset import get_train_transforms, get_val_transforms


def main(**kwargs):
    from_cli = OmegaConf.create(kwargs)
    base_conf = OmegaConf.load("./configs/_base_config.yaml")
    cfg = OmegaConf.merge(base_conf, from_cli)

    organ = cfg.organ.split("_")[1].lower()
    print(f"Training for {organ}")

    # Create dataset and dataloader
    image_files_train = sorted(
        [
            Path(cfg.data_2d_path, "train_2d_images", f).as_posix()
            for f in Path(cfg.data_2d_path, "train_2d_images").iterdir()
            if f.is_file() and f.stem.startswith(organ)
        ]
    )
    mask_files_train = sorted(
        [
            Path(cfg.data_2d_path, "train_2d_masks", f).as_posix()
            for f in Path(cfg.data_2d_path, "train_2d_masks").iterdir()
            if f.is_file() and f.stem.startswith(organ)
        ]
    )

    print(f"Number of training images: {len(image_files_train)}")
    sys.exit()

    image_files_train = sorted(
        [
            os.path.join(images_dir_train, f)
            for f in os.listdir(images_dir_train)
            if f.startswith(ORGAN) and f.endswith(".nii.gz")
        ]
    )
    label_files_train = sorted(
        [
            os.path.join(labels_dir_train, f)
            for f in os.listdir(labels_dir_train)
            if f.startswith(ORGAN) and f.endswith(".nii.gz")
        ]
    )
    image_files_val = sorted(
        [
            os.path.join(images_dir_val, f)
            for f in os.listdir(images_dir_val)
            if f.startswith(ORGAN) and f.endswith(".nii.gz")
        ]
    )
    label_files_val = sorted(
        [
            os.path.join(labels_dir_val, f)
            for f in os.listdir(labels_dir_val)
            if f.startswith(ORGAN) and f.endswith(".nii.gz")
        ]
    )

    train_transforms = get_train_transforms(cfg.roi_size)
    val_transforms = get_val_transforms()

    train_files = data_dicts_train
    val_files = data_dicts_val
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=cfg.num_workers)
    train_loader = DataLoader(train_ds, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.num_workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.val_bs, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model definition
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


if __name__ == "__main__":
    fire.Fire(main)
