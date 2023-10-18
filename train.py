from dataset.dataset import (
    getLoader
)
from model import UNet
import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from monai.data import DataLoader
from sklearn.model_selection import train_test_split

from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import argparse


def train(trainLoader, model):
    model.train()
    #initialize loss
    loss_dice_ave = 0

    # epoch iterator
    epoch_iterator = tqdm(
        trainLoader, desc="Training", dynamic_ncols=True
    )
    i = 0
    for step, batch in enumerate(epoch_iterator):
        if i > 0:
            break
        img, lbl, name = batch["image"], batch["label"].float(), batch['name']
        
        img = img[0][0] 
        lbl = lbl[0][0]
        img.cpu()
        lbl.cpu()
        print(f"image shape: {img.shape}, label shape: {lbl.shape}")
        # plot the slice [:, :, 80]
        plt.figure("check", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(img[:, :, 80], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(lbl[:, :, 80])
        plt.savefig('visualize_transformed_data.png')
        i+=1
    

def main():
    #parser = argparse.ArgumentParser()
    #args = parser.parse_args()

    # create the training loader
    trainLoader = getLoader('train', 'Task03_Liver')
    model = UNet()

    # call training loop
    train(trainLoader, model)
    

if __name__ == "__main__":
    main()