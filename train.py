from dataset.dataset import (
    getLoader
)
from model import UNet
import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from monai.data import DataLoader
from monai.losses import DiceCELoss
from sklearn.model_selection import train_test_split

from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import argparse

from utils.utils import (
    plotLoss,
    visualizeTransformedData,
    visualizeSegmentation,
    saveCheckpoint,
    loadCheckpoint
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(trainLoader, valLoader, model, optimizer, lossFunc):
    print('[INFO] started training the network...')

    train_losses = []
    val_losses = []

    # loop through epochs
    epoch_loop = tqdm(range(config.NUM_EPOCHS), desc='Epoch')
    for epoch in epoch_loop:
        epoch_loop.set_description(f"Epoch {epoch+1}")

        model.train()   #model in training mode

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # loop through the training set
        train_loop = tqdm(trainLoader, desc='Batch')
        for step, batch in enumerate(train_loop):
            train_loop.set_description(f"Batch {step+1}")
            img = batch["image"].to(config.DEVICE)
            lbl = batch["label"].float().to(config.DEVICE)
            name = batch['name']
            
            # see the first image (crop) 
            if step==0:
                visualizeTransformedData(img[0][0].to('cpu'),lbl[0][0].to('cpu'),60)

            # forward pass
            with torch.cuda.amp.autocast():
                predicted = model(img)
                loss = lossFunc(predicted, lbl)

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalTrainLoss += loss.item()
            
            # Save loss every 10th step
            if (step + 1) % 10 == 0:
                avg_loss = totalTrainLoss / 10
                print(f"\n Epoch {epoch + 1}, Train batch {step + 1}, Loss: {avg_loss:.4f}\n")
                train_losses.append(avg_loss)
                totalTrainLoss = 0.0

            # get binary segmentation 
            predicted_prob = torch.sigmoid(predicted[0][0])
            predicted_label = (predicted_prob > config.THRESHOLD).astype(np.uint8)

            visualizeSegmentation(img, lbl, name, predicted_label)

            torch.cuda.empty_cache()
        
        # validation loop
        with torch.no_grad():
            model.eval()
            
            for step, batch in enumerate(valLoader):
                img = batch["image"].to(config.DEVICE)
                lbl = batch["label"].float().to(config.DEVICE)
                name = batch['name']

                predicted = model(img)
                loss = lossFunc(predicted, lbl)
                totalValLoss += loss.item()

                # Save loss every 10th step
                if (step + 1) % 10 == 0:
                    avg_loss = totalValLoss / 10
                    print(f"\n Epoch {epoch + 1}, Val batch {step + 1}, Loss: {avg_loss:.4f}\n")
                    val_losses.append(avg_loss)
                    totalValLoss = 0.0
    
    plotLoss(train_losses, title= "Training Loss")
    plotLoss(val_losses, fig_path='output/plots/validationloss.png', title='Validation Loss')

    # save model
    saveCheckpoint(model, 'unet_task03_liver.pth')
        
        
    
    
    

def main():
    #parser = argparse.ArgumentParser()
    #args = parser.parse_args()

    # create loaders
    trainLoader = getLoader('train', 'Task03_Liver')
    valLoader = getLoader('val', 'Task03_Liver')
    
    #initialize model
    model = UNet().to(config.DEVICE)

    # initialize loss function and optimizer
    lossFunc = DiceCELoss()
    optimizer = AdamW(model.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY)

    # call training loop
    train(trainLoader, valLoader, model, optimizer, lossFunc)

    #x = torch.randn((3, 1, 256, 256, 256))
    #x = x.to(config.DEVICE)
    #preds = model(x)
    #print(preds.shape)
    #print(x.shape)
    

if __name__ == "__main__":
    main()