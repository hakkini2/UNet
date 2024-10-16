import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def visualize_and_save_image(image, file_name, size=8, cmap="gray", alpha=None):
    plt.figure(figsize=(size, size))
    plt.imshow(image, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def visualize_slices(input_dir, output_dir, split, organ):
    image_input_dir = os.path.join(input_dir, f"{split}_images")
    mask_input_dir = os.path.join(input_dir, f"{split}_masks")

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([os.path.join(image_input_dir, f) for f in os.listdir(image_input_dir) if f.startswith(organ) and f.endswith(".nii.gz")])
    mask_files = sorted([os.path.join(mask_input_dir, f) for f in os.listdir(mask_input_dir) if f.startswith(organ) and f.endswith(".nii.gz")])

    print(len(image_files), len(mask_files))

    # Loop through all files in the input directory
    for img_fname, mask_fname in zip(image_files, mask_files):
        # Load the 2D image and mask
        img = nib.load(img_fname)
        img_data = img.get_fdata()
        mask = nib.load(mask_fname)
        mask_data = mask.get_fdata()

        # Check if the dimensions match
        if img_data.shape != mask_data.shape:
            print(f"Data and mask dimensions do not match for {img_fname.split('/')[-1]}. Skipping.")
            continue
        
        '''
        # Create a figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axs[0].imshow(img_data, cmap='gray')
        axs[0].set_title('Original Image')
        #axs[0].axis('off')
        
        # Binary mask
        axs[1].imshow(mask_data, cmap='gray')
        axs[1].set_title('Binary Mask')
        #axs[1].axis('off')
        
        # Mask overlayed on image
        axs[2].imshow(img_data, cmap='gray')
        axs[2].imshow(mask_data, cmap='jet', alpha=0.5)
        axs[2].set_title('Overlay')
        #axs[2].axis('off')
        
        # Save the figure
        fname = img_fname.split('/')[-1]
        output_filename = os.path.join(output_dir, f"{fname.replace('.nii.gz', '')}.png")
        plt.savefig(output_filename)
        print(f"output_filename: {output_filename}")
        plt.close(fig)
        '''
        fname = img_fname.split('/')[-1]
        output_filename_img = os.path.join(output_dir, f"{fname.replace('.nii.gz', '')}_orig.png")
        output_filename_mask = os.path.join(output_dir, f"{fname.replace('.nii.gz', '')}_mask.png")
        visualize_and_save_image(img_data, output_filename_img)
        visualize_and_save_image(mask_data, output_filename_mask)
        #print(f"Saved visualization for slice {fname}")
        #sys.exit()
                
    print("Processing completed.")

# Example usage
input_directory = '2d_data'
output_directory = 'viz_slices'

visualize_slices(input_directory, output_directory, split="train", organ="spleen")
