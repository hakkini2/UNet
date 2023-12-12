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
from samDataset import get_loader, get_point_prompt, center_of_mass_from_3d, averaged_center_of_mass
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
split = 'test'
all_plots_path = config.SAM_OUTPUT_PATH + split + '_images/' + 'all/'
best_plots_path = config.SAM_OUTPUT_PATH + split + '_images/' + 'top_best/'
worst_plots_path = config.SAM_OUTPUT_PATH + split + '_images/' + 'top_worst/'
dices_path = config.SAM_OUTPUT_PATH + split + '_images/' + 'dices/'
Path(all_plots_path).mkdir(parents=True, exist_ok=True)
Path(best_plots_path).mkdir(parents=True, exist_ok=True)
Path(worst_plots_path).mkdir(parents=True, exist_ok=True)
Path(dices_path).mkdir(parents=True, exist_ok=True)


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    # get pretrained SAM model - directly from Meta
    model = sam_model_registry['vit_h'](checkpoint=config.SAM_CHECKPOINT_PATH)
    model.to(config.DEVICE)
    predictor = SamPredictor(model)

    # get dataloader containing half of the training images
    loader = get_loader(organ=config.ORGAN, split=split)
    # get trainloader to get the same point prompt
    train_loader = get_loader(organ=config.ORGAN, split='train')
    prompt = averaged_center_of_mass(train_loader)

    # do mask prediction and collect the dice scores
    dices = predict_masks(loader, prompt, predictor)

    saveDices(dices, split=split)


def predict_masks(loader, prompt, predictor):
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

            # get point prompt - individual for each img
            #prompt = get_point_prompt(ground_truth_mask)
            input_point = np.array([[prompt[1], prompt[0]]])
            input_label = np.array([1])

            # get predicted masks
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            # CHOOSE THE FIRST MASK FROM MULTIMASK OUTPUT 
            mask = masks[0]
            score = scores[0]

            #print(f'Image {step+1}, mask {i+1}:')
            # evaluate with dice score
            dice_pytorch = dice(torch.Tensor(mask).cpu(), ground_truth_mask.cpu(), ignore_index=0)
            dice_utils, _, _ = calculate_dice_score(torch.Tensor(mask).cpu(), ground_truth_mask.cpu())
            
		    # check that different dice scores match
            if not np.isclose(dice_pytorch, dice_utils.item()):
                print("DIFFERENT DICES \n")
                print(f"i: {step}, name: {name[0]}")
                break

            dices.append((name[0], dice_pytorch))
            
            print('dice pytorch: ', dice_pytorch)
            print('dice utils: ', dice_utils)

            plt.figure(figsize=(12, 4))
            plt.suptitle(f"{name[0]}, Score: {score:.3f}, Dice: {dice_pytorch:.3f}", fontsize=14)
            plt.subplot(1, 3, 1)
            plt.imshow(color_img, cmap="gray")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(color_img, cmap="gray")
            plt.imshow(ground_truth_mask.cpu().numpy(), alpha=0.6, cmap="copper")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(image_orig, cmap="gray")
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.axis("off")
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
        plt.savefig(f'{config.SAM_OUTPUT_PATH}{split}_images/{config.ORGAN}_dice_histogram.png')
        plt.close()

        return dices


def saveDices(dices, split):
    # save with pickle
    pickle_path = dices_path + f'{split}_dice_scores.pkl'
    with open (pickle_path, 'wb') as file:
        pickle.dump(dices, file)

    # reformat dice data for file
    dices = list(map(lambda dice: (dice[0], dice[1].item()), dices))
    
    # save to a file
    file_path = dices_path + f'{split}_dice_scores.txt'
    with open(file_path, 'w') as f:
        for line in dices:
            f.write(f'{line}\n')




if __name__ == '__main__':
    main()

