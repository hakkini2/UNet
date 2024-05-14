import nibabel as nib
import os
import numpy as np
from pathlib import Path
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


def compute_bounding_boxes(mask):
    """
    Compute bounding boxes for each connected component in the mask slice.
    :param mask_slice: A 2D numpy array of the mask
    :return: A list of bounding boxes in (min_row, min_col, max_row, max_col) format
    """
    # Label the connected components
    labeled_mask, num_features = ndi.label(mask)
    bounding_boxes = []

    for region in range(1, num_features + 1):  # Start from 1 as 0 is the background
        where = np.where(labeled_mask == region)
        min_row, max_row = np.min(where[0]), np.max(where[0])
        min_col, max_col = np.min(where[1]), np.max(where[1])
        bounding_boxes.append((min_row, min_col, max_row, max_col))

    return bounding_boxes


def visualize_nii():
    data_path = Path("/data2/projects/iira/UNet/content")
    organ = "lung"
    output_path = Path("output/raw_viz_bbox") / organ
    output_path.mkdir(parents=True, exist_ok=True)
    
    img_path = data_path / "val_2d_images"
    msk_path = data_path / "val_2d_masks"
    queries = [q_fname for q_fname in os.listdir(img_path) if q_fname[:len(organ)] == organ]
    
    for query_name in queries[:100]:
        image_nii = nib.load(img_path / query_name)
        mask_nii = nib.load(msk_path / query_name)
        
        # Get the image and mask data as numpy arrays
        image_data = image_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
        
        bounding_boxes = compute_bounding_boxes(mask_data)
        
        print(f"image_data shape: {image_data.shape}")
        print(f"mask_data shape: {mask_data.shape}")
        
        # Plot the image with the mask overlay
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_data, cmap='gray', origin='lower')
        ax.imshow(mask_data, cmap='jet', alpha=0.5, origin='lower')  # Adjust alpha for transparency
        
        for bbox in bounding_boxes:
            min_row, min_col, max_row, max_col = bbox
            rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                             edgecolor='green', facecolor='none', linewidth=2)
            ax.add_patch(rect)
        
        ax.axis('off')
        #fig.colorbar(label='Mask Intensity')  # Optional, shows intensity of the mask
        fig.savefig(f"{output_path}/fig-{query_name}.pdf", format='pdf', bbox_inches='tight')
        plt.close(fig)
        
    
visualize_nii()