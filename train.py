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


def train(trainLoader, model, optimizer, lossFunc):
    print('[INFO] started training the network...')

    # loop through epochs
    for epoch in tqdm(range(config.NUM_EPOCHS)):
        model.train()   #model in training mode

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0

        # loop through the training set
        for step, batch in enumerate(trainLoader):
            img = batch["image"].to(config.DEVICE)
            lbl = batch["label"].float().to(config.DEVICE)
            name = batch['name']

            # see the fist image 
            if step==0:
                visualizeTransformedData(img[0][0].to('cpu'),lbl[0][0].to('cpu'),200)

            print(img.shape)
            predicted = model(img)
            loss = lossFunc(predicted, lbl)

        
        
        
    
    

def visualizeTransformedData(img, lbl, slice_id):
    '''
    img and lbl should have only 3 channels, x,y,z
    '''
    print(f"image shape: {img.shape}, label shape: {lbl.shape}")

    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[:, :, slice_id], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(lbl[:, :, slice_id])
    plt.savefig('output/plots/visualize_transformed_data.png')


def main():
    #parser = argparse.ArgumentParser()
    #args = parser.parse_args()

    # create the training loader
    trainLoader = getLoader('train', 'Task03_Liver')
    
    #initialize model
    model = UNet().to(config.DEVICE)

    # initialize loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config.INIT_LR)

    # call training loop
    train(trainLoader, model, optimizer, lossFunc)
    

if __name__ == "__main__":
    main()