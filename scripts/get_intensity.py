import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Define paths and dataset-specific parameters
data_dir = "/data2/projects/iira/UNet/content/train_2d_images"
organ = "pancreas"

# Load file paths
image_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith(organ) and f.endswith(".nii.gz")])

# Initialize variables to store min and max intensity values
global_min = float('inf')
global_max = float('-inf')

print(f"The number of files: {len(image_files)}")

# Iterate through all training images
for image_file in tqdm(image_files):
    # Load the image
    img = nib.load(image_file)
    img_data = img.get_fdata()
    
    # Compute the min and max intensity values
    img_min = np.min(img_data)
    img_max = np.max(img_data)
    
    # Update global min and max
    if img_min < global_min:
        global_min = img_min
    if img_max > global_max:
        global_max = img_max

print(f"Global min intensity: {global_min}")
print(f"Global max intensity: {global_max}")
