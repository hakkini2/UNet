import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CropForegroundd,
    ScaleIntensityRanged,
    RandAffined,
    RandCropByPosNegLabeld,
    EnsureTyped,
)
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
import time


# config
SEED = 1984
NUM_WORKERS = 4
N_EPOCHS = 100
ROI_SIZE = 96
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
LR = 1e-2
STEP_LR_SIZE = 30
GAMMA_LR = 0.1
ORGAN = "spleen"
INTENSITIES = {"lung": (-1024, 5324),
               "colon": (-1024, 13009),
               #"spleen": (-1024, 3072),
               "spleen": (-57, 164),
               "liver": (-2048, 27572),
               "pancreas": (-2048, 3071),
               "hepaticvessel": (-1024, 3463)}

# Set random seed for reproducibility
set_determinism(seed=SEED)

torch.multiprocessing.set_sharing_strategy('file_system')

# Define paths and dataset-specific parameters
data_dir = "/data2/projects/iira/UNet/content"  # Change this to your dataset path

a_min, a_max = INTENSITIES[ORGAN]
images_dir_train = os.path.join(data_dir, "train_2d_images")
labels_dir_train = os.path.join(data_dir, "train_2d_masks")
images_dir_val = os.path.join(data_dir, "val_2d_images")
labels_dir_val = os.path.join(data_dir, "val_2d_masks")

# Load file paths
image_files_train = sorted([os.path.join(images_dir_train, f) for f in os.listdir(images_dir_train) if f.startswith(ORGAN) and f.endswith(".nii.gz")])
label_files_train = sorted([os.path.join(labels_dir_train, f) for f in os.listdir(labels_dir_train) if f.startswith(ORGAN) and f.endswith(".nii.gz")])
image_files_val = sorted([os.path.join(images_dir_val, f) for f in os.listdir(images_dir_val) if f.startswith(ORGAN) and f.endswith(".nii.gz")])
label_files_val = sorted([os.path.join(labels_dir_val, f) for f in os.listdir(labels_dir_val) if f.startswith(ORGAN) and f.endswith(".nii.gz")])

print(f"Organ: {ORGAN}")
print(f"The number of images/labels in train split: {len(image_files_train)}/{len(label_files_train)}")
print(f"The number of images/labels in val split: {len(image_files_val)}/{len(label_files_val)}")

# Create a list of dictionaries for the dataset
data_dicts_train = [{"image": image, "label": label} for image, label in zip(image_files_train, label_files_train)]
data_dicts_val = [{"image": image, "label": label} for image, label in zip(image_files_val, label_files_val)]


# Define transformations
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    #ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(ROI_SIZE, ROI_SIZE), pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0),
    RandAffined(keys=["image", "label"], prob=0.5, rotate_range=(0.1), shear_range=(0.1), translate_range=(10, 10), scale_range=(0.1), mode=("bilinear", "nearest")),
    EnsureTyped(keys=["image", "label"])
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    #ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image", "label"])
])

# Create dataset and dataloader
train_files = data_dicts_train
val_files = data_dicts_val
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, num_workers=NUM_WORKERS)

# Define network, loss, optimizer, and metric
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=STEP_LR_SIZE, gamma=GAMMA_LR)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Training loop
max_epochs = N_EPOCHS
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
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        if ORGAN in ["liver", "hepaticvessel", "pancreas"]:
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
        for val_data in val_loader:
            step += 1
            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            
            if ORGAN in ["liver", "hepaticvessel", "pancreas"]:
                val_labels[val_labels == 2] = 1
            
            val_outputs = sliding_window_inference(val_inputs, (ROI_SIZE, ROI_SIZE), 4, model)
            loss = loss_function(val_outputs, val_labels)
            epoch_loss_val += loss.item()
            
            # compute the metric
            val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
            val_labels = val_labels.long()
            val_outputs_list = decollate_batch(val_outputs)
            val_labels_list = decollate_batch(val_labels)
            dice_metric(y_pred=val_outputs_list, y=val_labels_list)
            epoch_loss_val += loss.item()

        epoch_loss_val /= step
        epoch_loss_val_values.append(epoch_loss_val)
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        metric_values.append(metric)
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), f"{ORGAN}_best_metric_model.pth")
            print("saved new best metric model")
        print(f"current epoch: {epoch + 1} current mean dice (val): {metric:.4f} best mean dice (val): {best_metric:.4f} at epoch: {best_metric_epoch}")
        print(f"Validation loss: {epoch_loss_val:.4f}")
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


# Plot loss vs. number of epochs
plt.figure()
plt.plot(range(1, len(epoch_loss_train_values) + 1), epoch_loss_train_values)
plt.title("Training Loss vs. Number of Epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.savefig(f"loss_vs_epochs_train_{ORGAN}.pdf")

plt.figure()
plt.plot(range(1, len(epoch_loss_val_values) + 1), epoch_loss_val_values)
plt.title("Training Loss vs. Number of Epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.savefig(f"loss_vs_epochs_val_{ORGAN}.pdf")

# Plot dice_metric vs. number of epochs
plt.figure()
plt.plot(range(1, len(metric_values) + 1), metric_values)
plt.title("Validation Dice Metric vs. Number of Epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("Dice Metric")
plt.savefig(f"dice_metric_vs_epochs_val_{ORGAN}.pdf")
