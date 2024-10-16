import os
import numpy as np
import nibabel as nib


def get_data_dicts(data_dir, txt_dir, split):
    list_img = []
    list_lbl = []
    list_name = []

    for line in open(txt_dir + '_' + split + '.txt'):
        name = line.strip().split()[1].split('.')[0]
        list_img.append(data_dir + line.strip().split()[0])
        list_lbl.append(data_dir + line.strip().split()[1])
        list_name.append(name)
    data_dicts = [{'image': image, 'label': label, 'name': name}
                for image, label, name in zip(list_img, list_lbl, list_name)]
    
    print(f"Split: {split}: the number of samples: {len(data_dicts)}")
    return data_dicts


def create_slice(output_dir, data_dict, split):
    image_output_dir = os.path.join(output_dir, f"{split}_images")
    mask_output_dir = os.path.join(output_dir, f"{split}_masks")
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # make slices
    for _, item in enumerate(data_dict):
        img_path = item['image']
        mask_path = item['label']

        # Load the 3D image and mask
        img = nib.load(img_path)
        img_data = img.get_fdata()
        img_affine = img.affine  # Get the affine matrix for the image

        mask = nib.load(mask_path)
        mask_data = mask.get_fdata()
        mask_affine = mask.affine  # Get the affine matrix for the mask

        # Check if the dimensions match
        if img_data.shape != mask_data.shape:
            print(f"Data and mask dimensions do not match for {img_path.split('/')[-1]}. Skipping.")
            continue

        # Iterate through the slices
        for i in range(img_data.shape[-1]):
            if np.any(mask_data[:, :, i]):
                image_slice = img_data[:, :, i]
                label_slice = mask_data[:, :, i]
                
                # Create new NIfTI images for the slices
                image_slice_nifti = nib.Nifti1Image(image_slice, img_affine)
                label_slice_nifti = nib.Nifti1Image(label_slice, mask_affine)
                
                # Save the NIfTI images
                image_filename = os.path.join(image_output_dir, f"{os.path.basename(img_path).replace('.nii.gz', '')}_slice_{i:03d}.nii.gz")
                mask_filename = os.path.join(mask_output_dir, f"{os.path.basename(mask_path).replace('.nii.gz', '')}_slice_{i:03d}.nii.gz")
                nib.save(image_slice_nifti, image_filename)
                nib.save(label_slice_nifti, mask_filename)

    print(f"{split}: processing completed.")


home_dir = "/data2/projects/CLIP-Driven-Universal-Model"
data_dir = os.path.join(home_dir, "data/MSD/") 
txt_dir = os.path.join(home_dir, "dataset/dataset_list/PAOT_10_inner")
output_dir = os.path.join("2d_data")

data_dicts_train = get_data_dicts(data_dir, txt_dir, split="train")
data_dicts_val = get_data_dicts(data_dir, txt_dir, split="val")
data_dicts_test = get_data_dicts(data_dir, txt_dir, split="test")

create_slice(output_dir, data_dicts_train, "train")
create_slice(output_dir, data_dicts_val, "val")
create_slice(output_dir, data_dicts_test, "test")
