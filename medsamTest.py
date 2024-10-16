import torch
import nibabel as nib
from PIL import Image
import os
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from transformers import SamModel, SamProcessor
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from samDataset import compute_bounding_boxes
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    EnsureTyped,
)
from utils.utils import normalize8, calculate_dice_score
from torchmetrics.functional.classification import dice


def show_mask(mask, ax, random_color):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2))


torch.multiprocessing.set_sharing_strategy("file_system")

# PARAMS
DATA_DIR = "/data2/projects/iira/UNet/2d_data"
NUM_WORKERS = 4
BATCH_SIZE = 1
SPLIT = 'test' # train, val, test
ORGAN = "spleen"

SAM_OUTPUT_PATH = "medsam-inference"
SAM_PROMPT = "box"

output_base_path = SAM_OUTPUT_PATH + "/" + SAM_PROMPT + '_prompt/' + SPLIT + '_images/'
all_plots_path = output_base_path + 'all/'
best_plots_path = output_base_path + 'top_best/'
worst_plots_path = output_base_path + 'top_worst/'
dices_path = output_base_path + 'dices/'
Path(output_base_path).mkdir(parents=True, exist_ok=True)
Path(all_plots_path).mkdir(parents=True, exist_ok=True)
Path(best_plots_path).mkdir(parents=True, exist_ok=True)
Path(worst_plots_path).mkdir(parents=True, exist_ok=True)
Path(dices_path).mkdir(parents=True, exist_ok=True)

images_dir_test = os.path.join(DATA_DIR, "test_images")
labels_dir_test = os.path.join(DATA_DIR, "test_masks")
# Load file paths
image_files_test = sorted([os.path.join(images_dir_test, f) for f in os.listdir(images_dir_test) if f.startswith(ORGAN) and f.endswith(".nii.gz")])
label_files_test = sorted([os.path.join(labels_dir_test, f) for f in os.listdir(labels_dir_test) if f.startswith(ORGAN) and f.endswith(".nii.gz")])
# Create a list of dictionaries for the dataset
data_dicts = [{"image": image, "label": label, "name": image.split('/')[-1][:-7], "fname": image} for image, label in zip(image_files_test, label_files_test)]

sam_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"])
])

#dataset = CacheDataset(data=data_dicts, transform=sam_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
dataset = Dataset(data=data_dicts, transform=sam_transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device)

with torch.no_grad():
    dice_vals = []
    for step, item in enumerate(loader):
        print(f'Step {step+1}/{len(loader)}')
        image_orig = item['image'].squeeze()
        ground_truth_mask = item['label'].squeeze().to(bool)
        name = item['name']
        fname = item['fname'][0]
        '''
        # process a nii.gz file
        nii_img = nib.load(fname)
        nii_data = nii_img.get_fdata()
        # Normalize the 2D slice to the range 0-255
        slice_2d_normalized = (nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data)) * 255
        slice_2d_normalized = slice_2d_normalized.astype(np.uint8)
        # Convert the normalized 2D slice to a PIL image
        # Stack the normalized slice to create an RGB image
        slice_rgb = np.stack((slice_2d_normalized,)*3, axis=-1)

        # Convert the RGB slice to a PIL image
        pil_image = Image.fromarray(slice_rgb) #.convert('RGB')
        #print(pil_image.size)
        '''
        

        # convert to rbg uint8 image
        color_img = cv2.cvtColor(image_orig.numpy(), cv2.COLOR_GRAY2RGB)
        color_img = normalize8(color_img)
        #print(color_img.shape)
        #sys.exit()
        
        # Predict using bounding box
        if SAM_PROMPT == 'box':
            box_prompt_list = compute_bounding_boxes(ground_truth_mask)
            # prepare image + box prompt for the model
            # TODO
            inputs = processor(color_img, input_boxes=[box_prompt_list[0]], return_tensors="pt").to(device)

        outputs = model(**inputs, multimask_output=False)
        medsam_probs = processor.image_processor.post_process_masks(outputs.pred_masks.sigmoid().cpu(),
                                                             inputs["original_sizes"].cpu(),
                                                             inputs["reshaped_input_sizes"].cpu(),
                                                             binarize=False)
        '''
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(np.array(pil_image))
        show_box(box_prompt_list[0], ax[0])
        ax[0].set_title("Input Image and Bounding Box")
        ax[0].axis("off")
        ax[1].imshow(np.array(pil_image))
        show_mask(mask=medsam_probs[0][0].cpu().numpy() > 0.5, ax=ax[1], random_color=False)
        show_box(box_prompt_list[0], ax[1])
        ax[1].set_title("MedSAM Segmentation")
        ax[1].axis("off")
        
        plt.savefig("Medsam-test.png")
        '''
        
        
        #print(inputs["original_sizes"])
        #print(probs.shape)
        #sys.exit()
        '''
        # apply sigmoid
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        
        print(medsam_seg.shape, ground_truth_mask.shape)
        sys.exit()
        '''
        #print(medsam_probs[0].shape, ground_truth_mask.shape)
        medsam_seg = (medsam_probs[0][0].cpu().numpy() > 0.5).astype(np.uint8)
        
        dice_val, _, _ = calculate_dice_score(torch.Tensor(medsam_seg).cpu(), ground_truth_mask.cpu())
        #dice_pytorch = dice(torch.Tensor(medsam_seg).cpu(), ground_truth_mask.cpu(), ignore_index=0)
        #print(dice_val, dice_pytorch)
        #sys.exit()
        dice_vals.append(dice_val.item())
        
        #print(f"dice_val: {dice_val}")
        #plt.close(fig)
        #sys.exit()
    avg = sum(dice_vals) / len(dice_vals)
    print(f'Average dice for organ {ORGAN}: {avg:.3f}')