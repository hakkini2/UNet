import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CropForegroundd,
    ScaleIntensityRanged,
    ScaleIntensityd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandFlipd,
    MapTransform,
    Transform,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandZoomd,
    EnsureTyped,
)
from monai.networks.nets import UNet, AttentionUnet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
from torchmetrics.classification import Dice
import time


# config
SEED = 1984
NUM_WORKERS = 4
N_EPOCHS = 200
ROI_SIZE = 384
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
#LR = 1e-4
LR = 1e-2
BINARY_THRESHOLD = 0.5
STEP_LR_SIZE = 50
GAMMA_LR = 0.1
ORGAN = "liver"
INTENSITIES = {"lung": (-1024, 5324),
               "colon": (-1024, 13009),
               "spleen": (-1024, 3072),
               "liver": (-2048, 27572),
               "pancreas": (-2048, 3071),
               "hepaticvessel": (-1024, 3463)}

# Set random seed for reproducibility
set_determinism(seed=SEED)

torch.multiprocessing.set_sharing_strategy('file_system')

# Define paths and dataset-specific parameters
#data_dir = "/data2/projects/iira/UNet/content"
data_dir = "/data2/projects/iira/UNet/2d_data"

a_min, a_max = INTENSITIES[ORGAN]
'''
images_dir_train = os.path.join(data_dir, "train_2d_images")
labels_dir_train = os.path.join(data_dir, "train_2d_masks")
images_dir_val = os.path.join(data_dir, "val_2d_images")
labels_dir_val = os.path.join(data_dir, "val_2d_masks")
'''
images_dir_train = os.path.join(data_dir, "train_images")
labels_dir_train = os.path.join(data_dir, "train_masks")
images_dir_val = os.path.join(data_dir, "val_images")
labels_dir_val = os.path.join(data_dir, "val_masks")


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


# Function to save training batch images
def save_training_batch(inputs, labels, epoch, batch_idx):
    batch_size = inputs.shape[0]
    for i in range(batch_size):
        image = inputs[i].cpu().numpy().squeeze()
        label = labels[i].cpu().numpy().squeeze()
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Image")
        ax[1].imshow(label, cmap='gray')
        ax[1].set_title("Label")
        ax[2].imshow(image, cmap='gray')
        ax[2].imshow(label, cmap='jet', alpha=0.5)  # Using alpha for transparency
        ax[2].set_title("Overlay")
        
        plt.savefig(os.path.join(f"epoch_{epoch+1}_batch_{batch_idx+1}_image_{i+1}.png"))
        plt.close(fig)


# Custom cropping transform to ensure foreground pixels are included
class RandCropForegroundd(Transform):
    def __init__(self, spatial_size, max_attempts=10, num_samples=4):
        self.spatial_size = spatial_size
        self.max_attempts = max_attempts
        self.num_samples = num_samples

    def __call__(self, data):
        for _ in range(self.max_attempts):
            cropped_list = RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label", spatial_size=self.spatial_size, pos=1, neg=1, num_samples=self.num_samples
            )(data)
            valid_crops = [crop for crop in cropped_list if crop["label"].sum() > 0]  # Ensure there are foreground pixels
            if len(valid_crops) == self.num_samples:
                return valid_crops
        return cropped_list

'''
# Custom cropping transform to ensure foreground pixels are included
class RandCropForegroundd(MapTransform):
    def __init__(self, keys, spatial_size, max_attempts=10, num_samples=4):
        super().__init__(keys)
        self.spatial_size = spatial_size
        self.max_attempts = max_attempts
        self.num_samples = num_samples
        self.cropper = RandCropByPosNegLabeld(
            keys=keys, label_key="label", spatial_size=spatial_size, pos=1, neg=0, num_samples=num_samples
        )

    def __call__(self, data):
        for _ in range(self.max_attempts):
            cropped = self.cropper(data)
            valid_crops = [cropped_item for cropped_item in cropped if cropped_item["label"].sum() > 0]
            if len(valid_crops) == self.num_samples:
                return cropped
        return self.cropper(data)  # Return the last attempt even if it has no foreground pixels
'''

# Define transformations
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    #ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    #ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityd(keys=["image", "label"]),
    #CropForegroundd(keys=["image", "label"], source_key="image"),
    #RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(ROI_SIZE, ROI_SIZE), pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0),
    #RandCropForegroundd(keys=["image", "label"], spatial_size=(ROI_SIZE, ROI_SIZE)),
    RandCropForegroundd(spatial_size=(ROI_SIZE, ROI_SIZE)),
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandAffined(keys=["image", "label"], prob=0.5, rotate_range=(0.1, 0.1), scale_range=(0.1, 0.1), mode=("bilinear", "nearest")),
    RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.9, max_zoom=1.1, mode=("bilinear", "nearest")),
    RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.1),
    RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.9, 1.1)),
    EnsureTyped(keys=["image", "label"])
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    #ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    #ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityd(keys=["image", "label"]),
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

'''
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(64, 128, 256, 512),
    strides=(2, 2, 2),
    num_res_units=2,
    dropout=0.1  # Added dropout to prevent overfitting
).to(device)
'''
'''
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    dropout=0.1  # Added dropout to prevent overfitting
).to(device)

'''
'''
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2,
    dropout=0.1  # Added dropout to prevent overfitting
).to(device)
'''

# The baseline architecture
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    dropout=0.1  # Added dropout to prevent overfitting
).to(device)
'''

# Load a pre-trained UNet model
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,  # Change to 1 for binary segmentation
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
    #norm=Norm.BATCH  # Use BatchNorm
).to(device)
'''
'''
model_dict = model.state_dict()
pretrained_dict = torch.hub.load('Project-MONAI/MONAI', 'unet', pretrained=True).state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.to(device)
# Modify the final layer to match the number of output channels
model.out_conv = torch.nn.Conv2d(128, 1, kernel_size=1).to(device)
'''

'''
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)


model = UNet(
    spatial_dims=2,  # Changed to 2 for 2D images
    in_channels=1,
    out_channels=1,  # Single output channel for binary segmentation
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
    dropout=0.1  # Added dropout to prevent overfitting
).to(device)
'''

loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
#loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True)
#loss_function = DiceFocalLoss(to_onehot_y=False, sigmoid=True)  # Use sigmoid activation for binary segmentation
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=STEP_LR_SIZE, gamma=GAMMA_LR)
monai_dice_metric = DiceMetric(include_background=False, reduction="mean")

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
    for batch_idx, batch_data in enumerate(train_loader):
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

         # Save the current training batch
        #save_training_batch(inputs, labels, epoch, batch_idx)
        #sys.exit()

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
            
            if ORGAN in ["liver", "hepaticvessel", "pancreas"]:
                val_labels[val_labels == 2] = 1
            
            #val_outputs = sliding_window_inference(val_inputs, (ROI_SIZE, ROI_SIZE), 4, model)
            #val_outputs = model(val_inputs)
            val_outputs = sliding_window_inference(val_inputs, (ROI_SIZE, ROI_SIZE), 4, model)
            val_outputs_loss = torch.stack([i for i in decollate_batch(val_outputs)], dim=0).to(device)
            loss = loss_function(val_outputs_loss, val_labels)
            epoch_loss_val += loss.item()
            
            '''
            val_outputs = torch.sigmoid(val_outputs)  # Apply sigmoid activation
            val_outputs = (val_outputs > BINARY_THRESHOLD).long()
            '''
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

            #print(val_labels)
            #print(66 * '*')
            #print(val_outputs)
            #sys.exit()

            
            # compute the metric
            #val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
            #val_labels = val_labels.long()
            '''
            val_outputs_list = decollate_batch(val_outputs)
            val_labels_list = decollate_batch(val_labels)
            # Ensure both prediction and labels are binary
            for i in range(len(val_outputs_list)):
                val_outputs_list[i] = val_outputs_list[i].long()
                val_labels_list[i] = val_labels_list[i].long()
            monai_dice_metric(y_pred=val_outputs_list, y=val_labels_list)
            '''
            monai_dice_metric(y_pred=val_outputs, y=val_labels)
            
            #tm_dice_metric(val_outputs, val_labels)
            #epoch_loss_val += loss.item()

        epoch_loss_val /= step
        epoch_loss_val_values.append(epoch_loss_val)
        metric = monai_dice_metric.aggregate().item()
        monai_dice_metric.reset()
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
