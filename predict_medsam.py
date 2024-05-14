import torch
import torch.nn.functional as F
import nibabel as nib
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from transformers import SamModel, SamProcessor
from PIL import Image
from datasets import load_dataset

import torchvision.transforms as T


def load_medsam_model(device="cuda"):
    model_id = "wanglab/medsam-vit-base"
    model = SamModel.from_pretrained(model_id).to(device)
    model.eval()
    processor = SamProcessor.from_pretrained(model_id)
    return model, processor

'''
def preprocess_input(img, bbox, size=(224, 224)):
    crop = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    crop = crop - np.min(crop)
    crop_normalized = crop / np.max(crop)
    pil_img = Image.fromarray((crop_normalized * 255).astype(np.uint8))

    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229])
    ])
    return transform(pil_img).unsqueeze(0)

    return pil_img
'''


def predict_mask(model, processor, img, bbox):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = img.shape
    img_rgb = np.stack([img] * 3, axis=-1)
    img_rgb = img_rgb - np.min(img_rgb)
    crop_normalized = img_rgb / np.max(img_rgb)
    pil_img = Image.fromarray((crop_normalized * 255).astype(np.uint8))
    
    inputs = processor(images=pil_img, input_boxes=[[bbox]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
        
    # apply sigmoid
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    # match input image
    medsam_seg_prob = F.interpolate(medsam_seg_prob.unsqueeze(0), size=(1, h, w), mode="nearest-exact")    
    # convert soft mask to hard mask
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

    return medsam_seg


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
    return iou


def compute_bounding_boxes(mask_slice):
    labeled_mask, num_features = ndi.label(mask_slice)
    bounding_boxes = []
    for region in range(1, num_features + 1):
        where = np.where(labeled_mask == region)
        x_min, x_max = np.min(where[1]), np.max(where[1])
        y_min, y_max = np.min(where[0]), np.max(where[0])
        
        #x_min = max(0, x_min - np.random.randint(0, perturb_px))
        #x_max = min(w, x_max + np.random.randint(0, perturb_px))
        #y_min = max(0, y_min - np.random.randint(0, perturb_px))
        #y_max = min(h, y_max + np.random.randint(0, perturb_px))
        
        bounding_boxes.append([x_min, y_min, x_max, y_max])
        '''
        min_row, max_row = np.min(where[0]), np.max(where[0])
        min_col, max_col = np.min(where[1]), np.max(where[1])
        
        # add perturbation to bounding box coordinates
        min_row = float(max(0, min_row - np.random.randint(0, perturb_px)))
        max_row = float(min(w, max_row + np.random.randint(0, perturb_px)))
        min_col = float(max(0, min_col - np.random.randint(0, perturb_px)))
        max_col = float(min(h, max_col + np.random.randint(0, perturb_px)))
        bounding_boxes.append([min_row, min_col, max_row, max_col])
        '''
    return bounding_boxes


def visualize_nii_sam():
    data_path = Path("/data2/projects/iira/UNet/content")
    organ = "lung"
    output_path = Path("output/sam_viz_bbox_based") / organ
    output_path.mkdir(parents=True, exist_ok=True)
    
    img_path = data_path / "test_2d_images"
    msk_path = data_path / "test_2d_masks"
    queries = [q_fname for q_fname in os.listdir(img_path) if q_fname[:len(organ)] == organ]
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, processor = load_medsam_model(device=device)
    
    iou_scores = []
    for query_name in queries:
        image_nii = nib.load(img_path / query_name)
        mask_nii = nib.load(msk_path / query_name)
        
        # Get the image and mask data as numpy arrays
        image_data = image_nii.get_fdata()
        mask_gt = mask_nii.get_fdata()
        
        bounding_boxes = compute_bounding_boxes(mask_gt)
        
        '''
        # Plot the image with the mask overlay
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_data, cmap='gray', origin='lower')
        ax.imshow(mask_gt, alpha=0.6, cmap="copper")
        
        for bbox in bounding_boxes:
            min_row, min_col, max_row, max_col = bbox
            rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                             edgecolor='green', facecolor='none', linewidth=1)
            ax.add_patch(rect)
        '''
        iou_scores_per_organ = []
        for bbox in bounding_boxes:
            predicted_mask = predict_mask(model, processor, image_data, bbox)
            #predicted_mask = T.Resize(mask_data.shape)(predicted_mask_logits[0][0].cpu())
            #predicted_mask = predicted_mask.numpy() > 0.5  # Thresholding
            
            
            #print(mask_data.shape, predicted_mask.shape)
            #sys.exit()
            
            # Compute IoU
            #gt_mask_region = mask_data[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            #pred_mask_region = predicted_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            iou_score = compute_iou(predicted_mask, mask_gt)
            print(f"query_name: {query_name}, iou_score: {iou_score}")
            #iou_scores.append(iou_score)
            
            # Overlay resized predicted mask
            #ax.imshow(predicted_mask.T, cmap='hot', alpha=0.5, origin='lower')#, extent=(bbox[1], bbox[3], bbox[0], bbox[2]))
            #ax.imshow(predicted_mask, cmap='bone', alpha=0.6)#, extent=(bbox[1], bbox[3], bbox[0], bbox[2]))
            iou_scores_per_organ.append(iou_score)

        iou_scores.append(np.mean(iou_scores_per_organ))

        #miou = np.mean(iou_scores)
        #ax.set_title(f'SAM (bbox-based), mIoU: {miou}')
        
        #ax.axis('off')
        #fig.colorbar(label='Mask Intensity')  # Optional, shows intensity of the mask
        #fig.savefig(f"{output_path}/fig-{query_name}-1.pdf", format='pdf', bbox_inches='tight')
        #plt.close(fig)
    print(23 * '*')
    print(f"The mIoU ({organ}): {np.mean(iou_scores)}")
    
visualize_nii_sam()