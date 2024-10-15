import argparse
import shutil
import tqdm
import sys
import SimpleITK as sitk
from pathlib import Path
import pickle
import nibabel as nib
import subprocess
import matplotlib.pyplot as plt
import monai
import cv2
import numpy as np
import pickle
import torch
from PIL import Image
from samDataset import (
    get_loader,
    get_point_prompt,
    center_of_mass_from_3d,
    averaged_center_of_mass,
    compute_center_of_mass,
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
from utils.prompts import (
	get_point_prompt_prediction,
	get_box_prompt_prediction,
	get_box_with_points_prediction,
	get_box_and_point_prompt_prediction,
    get_box_then_point_prompt_prediction
)
import config

split = 'train'


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    # get pretrained SAM model - directly from Meta
    model = sam_model_registry['vit_h'](checkpoint=config.SAM_CHECKPOINT_PATH)
    model.to(config.DEVICE)
    predictor = SamPredictor(model)

    loader = get_loader(organ=config.ORGAN, split=split)

    # do mask prediction
    predict_masks(loader, predictor)



def predict_masks(loader, predictor):

    # create directory for saving the pseudomasks
    pseudo_masks_path = config.DATASET_PATH_2D + split + '_2d_' + config.SAM_PROMPT + '_pseudomasks/'
    histograms_path = config.SAM_OUTPUT_PATH + config.SAM_PROMPT + '_prompt/' + split + '_pseudomask_histograms/'
    Path(pseudo_masks_path).mkdir(parents=True, exist_ok=True)
    Path(histograms_path).mkdir(parents=True, exist_ok=True)

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

            # process image to get image embedding
            predictor.set_image(color_img)
            
            # Predict using point prompts
            if config.SAM_PROMPT == 'point' or config.SAM_PROMPT == 'naive_point' or config.SAM_PROMPT == 'furthest_from_edges_point':
                mask, input_points, input_labels = get_point_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    point_type=config.SAM_PROMPT
                )

            # Predict using bounding box
            if config.SAM_PROMPT == 'box':
                mask, input_boxes = get_box_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor
                )
            
            # Predict using one bounding box and cluster points
            if config.SAM_PROMPT == 'one_box_with_points':
                mask, input_boxes, input_points, input_labels = get_box_with_points_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor
                )
            
            # Predict using a box and a point per cluster
            if config.SAM_PROMPT == 'box_and_point':
                mask, input_boxes, input_points, input_labels = get_box_and_point_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    background_point=False
                )
            
            #Predict using a box and a background point per cluster
            if config.SAM_PROMPT == 'box_and_background_point':
                mask, input_boxes, input_points, input_labels = get_box_and_point_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    background_point=True
                )

            #Predict using a box, then based on the box prediction choose a point per cluster
            if config.SAM_PROMPT == 'box_and_then_point':
                mask, input_boxes, input_points, input_labels = get_box_then_point_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    point_type='fg'
                )
            
            #Predict using a box, then based on the box prediction choose a point per cluster
            if config.SAM_PROMPT == 'box_and_then_background_point':
                mask, input_boxes, input_points, input_labels = get_box_then_point_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    point_type='bg'
                )

            #Predict using a box, then based on the box prediction choose a foreground/background point per cluster
            if config.SAM_PROMPT == 'box_and_then_fg/bg_point':
                mask, input_boxes, input_points, input_labels = get_box_then_point_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    point_type='fg/bg'
                )

            # needed for box prompt
            mask = np.squeeze(mask)

            # evaluate with dice score
            dice_pytorch = dice(torch.Tensor(mask).cpu(), ground_truth_mask.cpu(), ignore_index=0)
            dice_utils, _, _ = calculate_dice_score(torch.Tensor(mask).cpu(), ground_truth_mask.cpu())
            
		    # check that different dice scores match
            if not np.isclose(dice_pytorch, dice_utils.item()):
                print("DIFFERENT DICES \n")
                print(f"i: {step}, name: {name[0]}")
                break

            dices.append((name[0], dice_pytorch))

            # -- Save pseudomask --
            # modify mask datatype
            mask = mask.astype(np.float32)
            # save
            mask_img = nib.Nifti1Image(mask, affine=np.eye(4))
            nib.save(mask_img, f'{pseudo_masks_path}{name[0]}.nii')
            subprocess.run(['gzip', f'{pseudo_masks_path}{name[0]}.nii']) # compress

        # sort the dices by the dice score (highest first)
        dices.sort(key=lambda x: x[1].item(), reverse=True)

        # get average dice
        dice_values = list(map(lambda dice: dice[1].item(), dices))
        avg = sum(dice_values) / len(dice_values)
        print(f'Average dice for organ {config.ORGAN}: {avg:.3f}')
        
        # draw a histogram of dice scores
        plt.hist([dice_info[1].item() for dice_info in dices], bins=20)
        plt.title(f'SAM: {config.ORGAN} pseudomask dice histogram, avg: {avg:.3f}')
        noise = f'{"_noise" if config.USE_NOISE_FOR_BOX_PROMPT else ""}'; 
        plt.savefig(f'{histograms_path}{config.ORGAN}{noise}_pseudomask_dice_histogram.png')
        plt.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--organ',
                        default=config.ORGAN,
                        choices=config.ORGAN_LIST,
                        help = 'What organ to create labels for.'
                        )
    parser.add_argument('--prompt',
                        default=config.SAM_PROMPT,
                        choices=config.SAM_PROMPTS_LIST,
                        help = 'What SAM prompt to use.'
                        )
    args = parser.parse_args()
    
    config.ORGAN = args.organ
    config.SAM_PROMPT = args.prompt

    main()

