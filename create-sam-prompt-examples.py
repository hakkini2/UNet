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
    compute_bounding_boxes,
    compute_boxes_and_points,
    compute_boxes_and_background_points
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

from utils.prompts import get_box_then_point_prompt_prediction

import config

# create directories for saving plots
split = 'test' # train, val, test
output_base_path = config.SAM_OUTPUT_PATH + 'prompt_example_plots/'
Path(output_base_path).mkdir(parents=True, exist_ok=True)

#specify a specific image:
use_specific_image = True
example_cases = ['colon_194_62', 'colon_122_24', 'liver_52_121', 'liver_5_393', 'liver_40_70',
                 'hepaticvessel_175_36', 'pancreas_246_53', 'pancreas_262_39', 'pancreas_279_45']
easy_cases = ['spleen_31_43']
#image_name = example_cases[-1]#example_cases[5]
image_names = np.array([example_cases[-1], example_cases[0], example_cases[4]])


organs = [i.split('_')[0] for i in image_names]

organs_map = dict(zip(['liver', 'lung', 'pancreas', 'hapaticvessel', 'spleen', 'colon'], config.ORGAN_LIST))

def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    # get pretrained SAM model
    model = sam_model_registry['vit_h'](checkpoint=config.SAM_CHECKPOINT_PATH)
    model.to('cpu')
    predictor = SamPredictor(model)

    loaders = [get_loader(organ=organs_map[organ], split=split) for organ in organs]
    predict_masks(loaders, predictor)




def predict_masks(loaders, predictor):
    n_columns = 6
    n_rows = len(loaders)
    with torch.no_grad():
        plt.figure(figsize=(4*n_columns, 4*n_rows))
        for image_id, loader in enumerate(loaders):
            for step, item in enumerate(loader):
                print(f'Step {step+1}/{len(loader)}')
                image_orig = item['image'].squeeze()
                ground_truth_mask = item['label'].squeeze().to(bool)
                name = item['name']

                # skip all images except the specified one, if a specific image is desired
                # not efficient
                if use_specific_image and name[0] != image_names[image_id]:
                    continue

                #image_id = np.where(name[0] == image_names)[0][0]

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
                    point_type='furthest_from_edges_point'
                )

                # BOX PROMPT
                mask_box, input_boxes = get_box_prompt(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    noise=False
                )


                # NOISY BOX PROMPT
                mask_noisy_box, input_noisy_boxes = get_box_prompt(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    noise=True
                )

                # BOX + POINT PROMPT
                mask_box_and_point, box_and_point_boxes, box_and_point_points, box_and_point_labels = get_box_and_point_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    background_point=False
                )

                # BOX + POINT PROMPT
                mask_box_and_bg_point, box_and_bg_point_boxes, box_and_bg_point_points, box_and_bg_point_labels = get_box_and_point_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    background_point=True
                )

                # BOX AND THEN POINT PROMPT
                mask_box_and_then_point, box_and_then_point_boxes, box_and_then_point_points, box_and_then_point_labels = get_box_then_point_prompt_prediction(
                        ground_truth_mask=ground_truth_mask,
                        predictor=predictor,
                        point_type='fg/bg'
                    )

                # evaluate with dice score
                dice_naive_point = dice(torch.Tensor(mask_naive_point).cpu(), ground_truth_mask.cpu(), ignore_index=0)
                dice_point = dice(torch.Tensor(mask_point).cpu(), ground_truth_mask.cpu(), ignore_index=0)
                dice_approx_point = dice(torch.Tensor(mask_approx_point).cpu(), ground_truth_mask.cpu(), ignore_index=0)
                dice_box = dice(torch.Tensor(mask_box).cpu(), ground_truth_mask.cpu(), ignore_index=0)
                dice_noisy_box = dice(torch.Tensor(mask_noisy_box).cpu(), ground_truth_mask.cpu(), ignore_index=0)
                dice_box_and_point = dice(torch.Tensor(mask_box_and_point).cpu(), ground_truth_mask.cpu(), ignore_index=0)
                dice_box_and_bg_point = dice(torch.Tensor(mask_box_and_bg_point).cpu(), ground_truth_mask.cpu(), ignore_index=0)
                dice_box_and_then_point = dice(torch.Tensor(mask_box_and_then_point).cpu(), ground_truth_mask.cpu(), ignore_index=0)

                print(f'Dice with naive point prompt: {dice_naive_point:.3f}')
                print(f'Dice with point prompt: {dice_point:.3f}')
                print(f'Dice with point furthest from edges: {dice_approx_point:.3f}')
                print(f'Dice with box prompt: {dice_box:.3f}')
                print(f'Dice with noisy box prompt: {dice_box:.3f}')
                print(f'Dice with box and point prompt: {dice_box_and_point:.3f}')
                print(f'Dice with box and background point prompt: {dice_box_and_bg_point:.3f}')
                print(f'Dice with box and then point prompt: {dice_box_and_then_point:.3f}')

                font_scale = 1.0
                pad = 10

                offset = n_columns * image_id
                # ground truth

                print("Offset:", offset, "image_id", image_id)
                plt.subplot(n_rows, 6, 1 + offset)
                plt.text(-0.08,0.0,f'{organs[image_id].upper()}', weight='bold', fontsize=20, verticalalignment='bottom', rotation='vertical',transform=plt.gca().transAxes)

                if image_id == 0:
                    plt.text(0.5,1.025,f'GROUND TRUTH', weight='bold', fontsize=20, horizontalalignment='center',transform=plt.gca().transAxes)

                plt.imshow(color_img, cmap="gray")
                plt.imshow(ground_truth_mask.cpu().numpy(), alpha=0.6, cmap="copper")
                plt.axis("off")
                # naive point
                plt.subplot(n_rows, 6, 2 + offset)
                if image_id == 0:
                    plt.text(0.5,1.025,r'$\bf{POINT}_{\bf{CM}}$', weight='bold', fontsize=20, horizontalalignment='center',transform=plt.gca().transAxes)
                plt.imshow(image_orig, cmap="gray")
                show_mask(mask_naive_point, plt.gca())
                for input_naive_point, input_naive_label in zip(input_naive_points, input_naive_labels):
                    show_points(input_naive_point, input_naive_label, plt.gca())
                plt.text(
                    0.95,
                    0.95,
                    f'DSC={dice_naive_point:.3f}',
                    bbox={'facecolor': 'lightblue', 'pad': pad},
                    fontsize=16*font_scale,
                    ha='right',
                    va='top',
                    transform=plt.gca().transAxes
                )
                plt.axis("off")
                # point furthest from foreground edges
                plt.subplot(n_rows, 6, 3 + offset)
                if image_id == 0:
                    plt.text(0.5,1.025,r'$\bf{POINT}_{\bf{INTERIOR}}$', weight='bold', fontsize=20, horizontalalignment='center',transform=plt.gca().transAxes)

                plt.imshow(image_orig, cmap="gray")
                show_mask(mask_approx_point, plt.gca())
                for input_approx_point, input_approx_label in zip(input_approx_points, input_approx_labels):
                    show_points(input_approx_point, input_approx_label, plt.gca())
                plt.text(
                    0.95,
                    0.95,
                    f'DSC={dice_approx_point:.3f}',
                    bbox={'facecolor': 'lightblue', 'pad': pad},
                    fontsize=16*font_scale,
                    ha='right',
                    va='top',
                    transform=plt.gca().transAxes
                )
                plt.axis("off")
                # box
                plt.subplot(n_rows, 6, 4 + offset)
                if image_id == 0:
                    plt.text(0.5,1.025,r'$\bf{BOX}$', weight='bold', fontsize=20, horizontalalignment='center',transform=plt.gca().transAxes)
                plt.imshow(image_orig, cmap="gray")
                show_mask(mask_box, plt.gca())
                for input_box in input_boxes:
                    show_box(input_box, plt.gca())
                plt.text(
                    0.95,
                    0.95,
                    f'DSC={dice_box:.3f}',
                    bbox={'facecolor': 'lightblue', 'pad': pad},
                    fontsize=16*font_scale,
                    ha='right',
                    va='top',
                    transform=plt.gca().transAxes
                )
                plt.axis("off")
                # noisy box
                plt.subplot(n_rows, 6, 5 + offset)
                if image_id == 0:
                    plt.text(0.5,1.025,r'$\bf{BOX}_{\bf{NOISE}}$', weight='bold', fontsize=20, horizontalalignment='center',transform=plt.gca().transAxes)
                plt.imshow(image_orig, cmap="gray")
                show_mask(mask_noisy_box, plt.gca())
                for input_box in input_noisy_boxes:
                    show_box(input_box, plt.gca())
                plt.text(
                    0.95,
                    0.95,
                    f'DSC={dice_noisy_box:.3f}',
                    bbox={'facecolor': 'lightblue', 'pad': pad},
                    fontsize=16*font_scale,
                    ha='right',
                    va='top',
                    transform=plt.gca().transAxes
                )
                plt.axis("off")
                # Box and then background point
                plt.subplot(n_rows, 6, 6 + offset)
                if image_id == 0:
                    plt.text(0.5,1.025,r'$\bf{BOX}_{\bf{+PP/NP}}$', weight='bold', fontsize=20, horizontalalignment='center',transform=plt.gca().transAxes)

                plt.imshow(image_orig, cmap="gray")
                show_mask(mask_box_and_then_point, plt.gca())
                for point, label in zip(box_and_then_point_points, box_and_then_point_labels):
                    show_points(point, label, plt.gca())
                for input_box in box_and_then_point_boxes:
                    show_box(input_box, plt.gca())
                plt.text(
                    0.95,
                    0.95,
                    f'DSC={dice_box_and_then_point:.3f}',
                    bbox={'facecolor': 'lightblue', 'pad': pad},
                    fontsize=16*font_scale,
                    ha='right',
                    va='top',
                    transform=plt.gca().transAxes
                )
                plt.axis("off")
                
                break
        
        plt.axis("off")
        plt.subplots_adjust(hspace=1.0)
        plt.tight_layout()
        print(f"Saving plot to {output_base_path}{name[0]}-fig2sim.pdf")
        plt.savefig(f'{output_base_path}{name[0]}-fig2sim.pdf')
        plt.close()




# helper functions to get the prompts

def get_point_prompt(ground_truth_mask, predictor, point_type):
    if point_type == 'naive_point':
        center_of_mass_list = compute_center_of_mass_naive(ground_truth_mask)
    elif point_type == 'point':
        center_of_mass_list = compute_center_of_mass(ground_truth_mask)
    elif point_type == 'furthest_from_edges_point':
        center_of_mass_list, _ = compute_furthest_point_from_edges(ground_truth_mask)

    mask = np.full(ground_truth_mask.shape, False, dtype=bool)
    input_points = []
    input_labels = []

    #loop through centers of mass, get sam's predictions for all and construct the final mask
    for i, center_of_mass in enumerate(center_of_mass_list):
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


def get_box_prompt(ground_truth_mask, predictor, noise=False):
    box_prompt_list = compute_bounding_boxes(ground_truth_mask, noise)
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


def get_box_and_point_prompt_prediction(ground_truth_mask, predictor, background_point=False):
    '''
    Get SAM's prediction using a box and a point per cluster
    '''
    if background_point:
        box_prompt_list, point_prompt_list = compute_boxes_and_background_points(ground_truth_mask)
    else:
        box_prompt_list, point_prompt_list = compute_boxes_and_points(ground_truth_mask)

    mask = np.full(ground_truth_mask.shape, False, dtype=bool)
    input_points = []
    input_labels = []
    input_boxes = []

    for i, point_prompt in enumerate(point_prompt_list):
        #point
        input_point = np.array([[round(point_prompt[1]), round(point_prompt[0])]])
        if background_point:
            input_label = np.array([0])
        else:
            input_label =  np.array([1])
        input_points.append(input_point)
        input_labels.append(input_label)

        # box
        input_box = np.array(box_prompt_list[i])
        input_boxes.append(input_box)
    
        # get predicted mask
        cluster_mask, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )

        # add cluster mask to final mask
        mask = mask | cluster_mask
    
    return mask, input_boxes, input_points, input_labels



if __name__ == '__main__':
    main()

