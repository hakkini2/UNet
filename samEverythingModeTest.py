import argparse
import shutil
import tqdm
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import monai
import cv2
import numpy as np
import sys
import pickle
import torch
from PIL import Image
from samDataset import (
    get_loader,
    get_point_prompt,
    center_of_mass_from_3d,
    averaged_center_of_mass,
    compute_center_of_mass,
    compute_center_of_mass_naive,
    compute_furthest_point_from_edges,
    compute_bounding_boxes,
    compute_one_bounding_box,
    compute_boxes_and_points,
    compute_boxes_and_background_points
)
from torchmetrics.functional.classification import dice
from transformers import SamModel, SamProcessor

# imports directly from meta's codebase
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

from utils.utils import (
    calculate_dice_score,
    show_mask,
    show_points,
    show_box,
    show_anns,
    normalize8
)
from utils.prompts import (
	get_point_prompt_prediction,
	get_box_prompt_prediction,
	get_box_with_points_prediction,
	get_box_and_point_prompt_prediction,
    get_point_prompt,
    get_point_prompt_prediction_2,
    get_box_with_points_prediction,
)

import config

# create directories for saving plots
split = 'test' # train, val, test
output_base_path = config.SAM_OUTPUT_PATH + 'everything_mode/' + split + '_images/'
all_plots_path = output_base_path + 'all/'
best_plots_path = output_base_path + 'top_best/'
worst_plots_path = output_base_path + 'top_worst/'
dices_path = output_base_path + 'dices/'
Path(output_base_path).mkdir(parents=True, exist_ok=True)
Path(all_plots_path).mkdir(parents=True, exist_ok=True)
Path(best_plots_path).mkdir(parents=True, exist_ok=True)
Path(worst_plots_path).mkdir(parents=True, exist_ok=True)
Path(dices_path).mkdir(parents=True, exist_ok=True)


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    # get pretrained SAM model - directly from Meta
    model = sam_model_registry['vit_h'](checkpoint=config.SAM_CHECKPOINT_PATH)
    model.to(config.DEVICE)
    mask_generator = SamAutomaticMaskGenerator(model)

    # get dataloader
    loader = get_loader(organ=config.ORGAN, split=split)

    # do mask prediction and collect the dice scores
    predict_masks(loader, mask_generator)



def predict_masks(loader, mask_generator):
    print(f'Predictions using everything mode')

    dices = []
    with torch.no_grad():
        for step, item in enumerate(loader):
            print(f'Step {step+1}/{len(loader)}')
            image_orig = item['image'].squeeze()
            ground_truth_mask = item['label'].squeeze().to(bool)
            name = item['name']

            # convert to rbg uint8 image
            color_img = cv2.cvtColor(image_orig.numpy(), cv2.COLOR_GRAY2RGB)
            color_img = normalize8(color_img)

            # get masks with everything mode
            masks = mask_generator.generate(color_img)

            # choose mask with best dice
            best_mask = masks[0]['segmentation']
            best_dice = 0
            for mask in masks:
                segmentation = mask['segmentation']
                dice_pytorch = dice(torch.Tensor(segmentation).cpu(), ground_truth_mask.cpu(), ignore_index=0)
                if dice_pytorch >= best_dice:
                    best_dice = dice_pytorch
                    best_mask = mask

            dices.append((name[0], best_dice))
            print('best dice: ', best_dice)

            #Plot image, ground truth, and predictions
            plt.figure(figsize=(16, 4))
            plt.suptitle(f"{name[0]}, Dice: {best_dice:.3f}", fontsize=14)
            plt.subplot(1, 4, 1)
            plt.title('Input')
            plt.imshow(color_img, cmap="gray")
            plt.axis("off")
            #ground truth
            plt.subplot(1, 4, 2)
            plt.title('Ground truth')
            plt.imshow(color_img, cmap="gray")
            plt.imshow(ground_truth_mask.cpu().numpy(), alpha=0.6, cmap="copper")
            plt.axis("off")
            # all masks
            plt.subplot(1, 4, 3)
            plt.title('Everything mode')
            plt.imshow(color_img, cmap="gray")
            show_anns(masks)
            plt.axis("off")
            # best mask
            plt.subplot(1, 4, 4)
            plt.title('Best dice match')
            plt.imshow(image_orig, cmap='gray')
            show_anns([best_mask])
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{all_plots_path}sam_{name[0]}.png')
            plt.close()

        # sort the dices by the dice score (highest first)
        dices.sort(key=lambda x: x[1].item(), reverse=True)

        # top config.N_TEST_SAMPLES best and worst segmentation results
        top_best, top_worst = dices[: config.N_TEST_SAMPLES], dices[-config.N_TEST_SAMPLES :]

        # copy top N best and worst samples to the corresponding directories from all_plots_path 
        for fname, _ in top_best:
            shutil.copy(f'{all_plots_path}sam_{fname}.png', f'{best_plots_path}sam_{fname}.png')
        for fname, _ in top_worst:
            shutil.copy(f'{all_plots_path}sam_{fname}.png', f'{worst_plots_path}sam_{fname}.png')

        # get average dice
        dice_values = list(map(lambda dice: dice[1].item(), dices))
        avg = sum(dice_values) / len(dice_values)
        print(f'Average dice for organ {config.ORGAN}: {avg:.3f}')
        
        # draw a histogram of dice scores
        plt.hist([dice_info[1].item() for dice_info in dices], bins=20)
        plt.title(f'SAM: {config.ORGAN} dice histogram, avg: {avg:.3f}')
        plt.savefig(f'{output_base_path}{config.ORGAN}_dice_histogram.png')
        plt.close()





if __name__ == '__main__':
    main()

