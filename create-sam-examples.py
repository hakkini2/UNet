import argparse
import shutil
import tqdm
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import monai
import cv2
import numpy as np
import pickle
import torch
from PIL import Image
from samDataset import (
    get_loader,
    compute_center_of_mass_naive,
    compute_center_of_mass,
    compute_furthest_point_from_edges,
    compute_bounding_boxes
)
from torchmetrics.functional.classification import dice
from transformers import SamModel, SamProcessor

# imports directly from meta's codebase
from segment_anything import SamPredictor, sam_model_registry

from utils.utils import (
    calculate_dice_score,
    show_mask,
    show_points,
    show_box,
    normalize8
)
import config

# create directories for saving plots
split = 'test' # train, val, test
output_base_path = config.SAM_OUTPUT_PATH + 'prompt_example_plots/'
Path(output_base_path).mkdir(parents=True, exist_ok=True)

#specify a specific image:
use_specific_image = True
interesting_cases = ['colon_194_62', 'colon_122_24', 'liver_52_121']
image_name = 'spleen_31_43'


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    # get pretrained SAM model
    model = sam_model_registry['vit_h'](checkpoint=config.SAM_CHECKPOINT_PATH)
    model.to(config.DEVICE)
    predictor = SamPredictor(model)

    loader = get_loader(organ=config.ORGAN, split=split)
    predict_masks(loader, predictor)




def predict_masks(loader, predictor):
    with torch.no_grad():
        for step, item in enumerate(loader):
            print(f'Step {step+1}/{len(loader)}')
            image_orig = item['image'].squeeze()
            ground_truth_mask = item['label'].squeeze().to(bool)
            name = item['name']

            # skip all images except the specified one, if a specific image is desired
            # not efficient
            if use_specific_image and image_name != name[0]:
                continue

            # convert to rbg uint8 image
            color_img = cv2.cvtColor(image_orig.numpy(), cv2.COLOR_GRAY2RGB)
            color_img = normalize8(color_img)

            # process image to get image embedding
            predictor.set_image(color_img)


            # POINT PROMPT -NAIVE
            mask_naive_point, input_naive_points, input_naive_labels = get_point_prompt(
                ground_truth_mask=ground_truth_mask,
                predictor=predictor,
                point_type='naive_point'
            )

            # POINT PROMPT - cm issue fixed
            mask_point, input_points, input_labels = get_point_prompt(
                ground_truth_mask=ground_truth_mask,
                predictor=predictor,
                point_type='point'
            )

            # POINT PROMPT - find the point furthest from the foreground edges
            mask_approx_point, input_approx_points, input_approx_labels = get_point_prompt(
                ground_truth_mask=ground_truth_mask,
                predictor=predictor,
                point_type='approx_mask_center'
            )

            # BOX PROMPT
            mask_box, input_boxes = get_box_prompt(
                ground_truth_mask=ground_truth_mask,
                predictor=predictor
            )


            # evaluate with dice score
            dice_naive_point = dice(torch.Tensor(mask_naive_point).cpu(), ground_truth_mask.cpu(), ignore_index=0)
            dice_point = dice(torch.Tensor(mask_point).cpu(), ground_truth_mask.cpu(), ignore_index=0)
            dice_approx_point = dice(torch.Tensor(mask_approx_point).cpu(), ground_truth_mask.cpu(), ignore_index=0)
            dice_box = dice(torch.Tensor(mask_box).cpu(), ground_truth_mask.cpu(), ignore_index=0)

            print(f'Dice with naive point prompt: {dice_naive_point:.3f}')
            print(f'Dice with point prompt: {dice_point:.3f}')
            print(f'Dice with point furthest from edges: {dice_approx_point:.3f}')
            print(f'Dice with box prompt: {dice_box:.3f}')

            plt.figure(figsize=(12, 8))
            plt.suptitle(f"SAM prompts on {name[0].split('_')[0]}", fontsize=14)
            # ground truth
            plt.subplot(2, 3, 1)
            plt.title('Ground truth')
            plt.imshow(color_img, cmap="gray")
            plt.imshow(ground_truth_mask.cpu().numpy(), alpha=0.6, cmap="copper")
            plt.axis("off")
            # naive point
            plt.subplot(2, 3, 2)
            plt.title(f'Naive point prompt (DSC={dice_naive_point:.3f})')
            plt.imshow(image_orig, cmap="gray")
            show_mask(mask_naive_point, plt.gca())
            for input_naive_point, input_naive_label in zip(input_naive_points, input_naive_labels):
                show_points(input_naive_point, input_naive_label, plt.gca())
            plt.axis("off")
            # point
            plt.subplot(2, 3, 3)
            plt.title(f'Point prompt (DSC={dice_point:.3f})')
            plt.imshow(image_orig, cmap="gray")
            show_mask(mask_point, plt.gca())
            for input_point, input_label in zip(input_points, input_labels):
                show_points(input_point, input_label, plt.gca())
            plt.axis("off")
            # point furthest from foreground edges
            plt.subplot(2, 3, 4)
            plt.title(f'Furthest point from edges (DSC={dice_approx_point:.3f})')
            plt.imshow(image_orig, cmap="gray")
            show_mask(mask_approx_point, plt.gca())
            for input_approx_point, input_approx_label in zip(input_approx_points, input_approx_labels):
                show_points(input_approx_point, input_approx_label, plt.gca())
            plt.axis("off")
            # box
            plt.subplot(2, 3, 5)
            plt.title(f'Box prompt (DSC={dice_box:.3f})')
            plt.imshow(image_orig, cmap="gray")
            show_mask(mask_box, plt.gca())
            for input_box in input_boxes:
                show_box(input_box, plt.gca())
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f'{output_base_path}{name[0]}.png')
            plt.close()



# helper functions to get the prompts

def get_point_prompt(ground_truth_mask, predictor, point_type):
    if point_type == 'naive_point':
        center_of_mass_list = compute_center_of_mass_naive(ground_truth_mask)
    elif point_type == 'point':
        center_of_mass_list = compute_center_of_mass(ground_truth_mask)
    elif point_type == 'approx_mask_center':
        center_of_mass_list = compute_furthest_point_from_edges(ground_truth_mask)

    mask = np.full(ground_truth_mask.shape, False, dtype=bool)
    input_points = []
    input_labels = []

    #loop through centers of mass, get sam's predictions for all and construct the final mask
    for i, center_of_mass in enumerate(center_of_mass_list):
        #print(f"Center of mass for object {i + 1}: {center_of_mass}")
        input_point = np.array([[round(center_of_mass[1]), round(center_of_mass[0])]])
        input_label =  np.array([1])
        input_points.append(input_point)
        input_labels.append(input_label)
    
        # get predicted masks
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # CHOOSE THE FIRST MASK FROM MULTIMASK OUTPUT 
        cluster_mask = masks[0]

        # add cluster to final mask
        mask = mask | cluster_mask

    return mask, input_points, input_labels


def get_box_prompt(ground_truth_mask, predictor):
    box_prompt_list = compute_bounding_boxes(ground_truth_mask)
    mask = np.full(ground_truth_mask.shape, False, dtype=bool)
    input_boxes = []

    #loop through clusters, get sam's predictions for all and construct the final mask
    for i, box_prompt in enumerate(box_prompt_list):
        #create input prompt
        input_box = np.array(box_prompt)
        input_boxes.append(input_box)

        # get predicted masks
        cluster_mask, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        #add cluster to final mask
        mask = mask | cluster_mask
    
    return mask, input_boxes



if __name__ == '__main__':
    main()

