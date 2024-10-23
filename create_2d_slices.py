from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fire
import numpy as np
import SimpleITK as sitk
from omegaconf import OmegaConf
from tqdm import tqdm


def process_image(item, img_out_dir, mask_out_dir):
    """
    Processes one image and its slices, saving each slice as a separate file.
    """
    # get paths to image and label
    img_path = Path(item["image"])
    lbl_path = Path(item["label"])

    # load image and label
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(lbl_path)

    # Get the mask data as numpy array
    mask_data = sitk.GetArrayFromImage(mask)

    # iterate over the axial slices
    for i in range(img.GetSize()[2]):
        # If the mask slice is not empty, save the image and mask slices
        if np.any(mask_data[i, :, :]):
            # Prepare the new ITK images
            img_slice = img[:, :, i]
            mask_slice = mask[:, :, i]

            # Define the output paths
            img_slice_path = img_out_dir / f"{img_path.stem[:-4]}_{i}.nii.gz"
            mask_slice_path = mask_out_dir / f"{lbl_path.stem[:-4]}_{i}.nii.gz"

            # Save the slices as NIfTI files
            sitk.WriteImage(img_slice, img_slice_path)
            sitk.WriteImage(mask_slice, mask_slice_path)


def get_data_dicts(cfg, split="train"):
    list_img = []
    list_lbl = []
    list_name = []

    for line in open(f"{cfg.data_txt_path}_{split}.txt"):
        name = line.strip().split()[1].split(".")[0]
        list_img.append(cfg.msd_data_path + line.strip().split()[0])
        list_lbl.append(cfg.msd_data_path + line.strip().split()[1])
        list_name.append(name)
    data_dicts = [
        {"image": image, "label": label, "name": name} for image, label, name in zip(list_img, list_lbl, list_name)
    ]

    print(f"split: {split}, # of images: {len(data_dicts)}")
    return data_dicts


def make_2d_slices(cfg, split, data_dicts):
    """
    Creates2d images and 2d masks under cfg.data_2d_path
    """
    img_out_dir = Path(cfg.data_2d_path, f"{split}_2d_images")
    mask_out_dir = Path(cfg.data_2d_path, f"{split}_2d_masks")
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    # Parallel processing pool for images
    with ProcessPoolExecutor() as executor:
        futures = []
        for item in data_dicts:
            # Submit each image for parallel processing
            futures.append(executor.submit(process_image, item, img_out_dir, mask_out_dir))

        # Wait for all tasks to complete
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()


def main(**kwargs):
    from_cli = OmegaConf.create(kwargs)
    base_conf = OmegaConf.load("./configs/_base_config.yaml")
    cfg = OmegaConf.merge(base_conf, from_cli)

    for split in ["train", "val", "test"]:
        data_dicts = get_data_dicts(cfg, split=split)
        make_2d_slices(cfg, split, data_dicts)


if __name__ == "__main__":
    fire.Fire(main)
