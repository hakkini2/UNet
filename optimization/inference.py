import os
import monai
import torch
import numpy as np
import nibabel as nib
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, EnsureTyped
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
import matplotlib.pyplot as plt

# Define paths and dataset-specific parameters
data_dir = "/data2/projects/iira/UNet/content"  # Change this to your dataset path
organ = "spleen"  # Example task, change to desired organ task
images_dir = os.path.join(data_dir, "test_2d_images")
labels_dir = os.path.join(data_dir, "test_2d_masks")
output_dir = f"output_preds/{organ}"  # Change this to your desired output path
TEST_BATCH_SIZE = 64
INTENSITIES = {"lung": (-1024, 5324),
               "colon": (-1024, 13009),
               "spleen": (-1024, 3072),
               "liver": (-2048, 27572),
               "pancreas": (-2048, 3071),
               "hepaticvessel": (-1024, 3463)}

a_min, a_max = INTENSITIES[organ]

# Load file paths
image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.startswith(organ) and f.endswith(".nii.gz")])
label_files = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.startswith(organ) and f.endswith(".nii.gz")])

test_dicts = [{"image": image, "label": label} for image, label in zip(image_files, label_files)]

# Define transformations for the test data
test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image", "label"])
])

# Create dataset and dataloader
test_ds = Dataset(data=test_dicts, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, num_workers=4, drop_last=False)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
model.load_state_dict(torch.load(f"{organ}_best_metric_model.pth"))
model.eval()

# Inference function
def infer_and_visualize(test_loader, model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    roi_size = (96, 96)
    sw_batch_size = 4
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    with torch.no_grad():
        for idx, test_data in enumerate(test_loader):
            test_image = test_data["image"].to(device)
            test_label = test_data["label"].to(device)
            if organ in ["liver", "hepaticvessel", "pancreas"]:
                test_label[test_label == 2] = 1
            test_output = sliding_window_inference(test_image, roi_size, sw_batch_size, model)
            
            m_outputs = torch.argmax(test_output, dim=1, keepdim=True)
            m_labels = test_label.long()
            val_outputs_list = decollate_batch(m_outputs)
            val_labels_list = decollate_batch(m_labels)
            dice_metric(y_pred=val_outputs_list, y=val_labels_list)

            '''
            test_output = torch.argmax(test_output, dim=1).cpu().numpy()[0]
            
            # Save the output segmentation
            test_image_meta = nib.load(image_files[idx])
            test_output_nifti = nib.Nifti1Image(test_output.astype(np.uint8), test_image_meta.affine, test_image_meta.header)
            nib.save(test_output_nifti, os.path.join(output_dir, f"seg_{os.path.basename(image_files[idx])}"))
            print(f"Saved segmentation for {image_files[idx]}")

            # Visualize the results
            test_image_np = test_image.cpu().numpy()[0, 0, :, :]
            test_label_np = test_label.cpu().numpy()[0, :, :]
            plt.figure("Visualization", (12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(test_image_np.squeeze(), cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(test_label_np.squeeze(), cmap="gray")
            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(test_output, cmap="gray")
            plt.savefig(os.path.join(output_dir, f"vis_{os.path.basename(image_files[idx])}.png"))
            '''
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        print(f"The dice score: {metric}")
        
# Run inference and visualize results
infer_and_visualize(test_loader, model, output_dir)
