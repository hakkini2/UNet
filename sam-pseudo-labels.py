import argparse
import shutil
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import monai
import numpy as np
import torch
from PIL import Image
from samDataset import get_loader, get_point_prompt
from torchmetrics.functional.classification import dice
from transformers import SamModel, SamProcessor

# imports directly from meta's codebase
from segment_anything import SamPredictor, sam_model_registry

import config

def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    # get pretrained SAM model
    model = sam_model_registry['vit_h'](checkpoint=config.SAM_CHECKPOINT_PATH)
    model.to(config.DEVICE)
    predictor = SamPredictor(model)
    loader = get_loader(organ=config.ORGAN, split='train')


    for i, item in enumerate(loader):
        if i==1:
            break
        image_orig = item['image'].squeeze()
        ground_truth_mask = item['label'].squeeze().to(bool)

        # convert to int type
        image = image_orig.astype(np.uint8)

        # convert the grayscale array to RGB (3 channels)
        array_rgb = np.dstack((image, image, image))

        # convert to PIL image to match the expected input of processor
        image_rgb = Image.fromarray(array_rgb)

        prompt = get_point_prompt(ground_truth_mask)

        # check point prompt
        plt.figure()
        plt.title('point prompt')
        plt.imshow(image_orig, cmap="gray")
        plt.imshow(ground_truth_mask.cpu().numpy(), alpha=0.6, cmap="copper")
        plt.plot(prompt[1], prompt[0], 'ro')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig('point_prompt_example.png')
        plt.close()



if __name__ == '__main__':
    main()

