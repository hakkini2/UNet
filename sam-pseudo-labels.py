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
import config

# create directory for saving the pseudomasks
split = 'train'
pseudo_masks_path = config.DATASET_PATH_2D + split + '_2d_' + config.SAM_PROMPT + '_pseudomasks/'
histograms_path = config.SAM_OUTPUT_PATH + config.SAM_PROMPT + '_prompt/' + split + '_pseudomask_histograms/'
Path(pseudo_masks_path).mkdir(parents=True, exist_ok=True)
Path(histograms_path).mkdir(parents=True, exist_ok=True)

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
            


            # Predict using point prompt
            if config.SAM_PROMPT == 'point':

                # get list of point prompts - one for each cluster
                center_of_mass_list = compute_center_of_mass(ground_truth_mask)

                #initialize mask array 
                mask = np.full(ground_truth_mask.shape, False, dtype=bool)
                # initialize lists for input poitns and labels for plotting
                input_points = []
                input_labels = []

                #loop through centers of mass, get sam's predictions for all and construct the final mask
                for i, center_of_mass in enumerate(center_of_mass_list):
                    print(f"Center of mass for object {i + 1}: {center_of_mass}")
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
            


            # Predict using bounding box
            if config.SAM_PROMPT == 'box':
                #get a list of bounding boxes - one for each cluster
                box_prompt_list = compute_bounding_boxes(ground_truth_mask)

                #initialize mask array 
                mask = np.full(ground_truth_mask.shape, False, dtype=bool)
                # initialize lists for input boxes for plotting
                input_boxes = []

                #loop through clusters, get sam's predictions for all and construct the final mask
                for i, box_prompt in enumerate(box_prompt_list):
                    print(f'Bounding box for cluster {i+1}: bottom: ({box_prompt[0]}, {box_prompt[1]}), top: ({box_prompt[2]}, {box_prompt[3]})')

                    #create input pormpt
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
        plt.savefig(f'{histograms_path}{config.ORGAN}_pseudomask_dice_histogram.png')
        plt.close()





if __name__ == '__main__':
    main()

