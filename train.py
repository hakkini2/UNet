from dataset import (
    getLoader3d
)
from unet3d import UNet3D
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
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(trainLoader, valLoader, model, optimizer, lossFunc):
    print('[INFO] started training the network...')

    # initialize lists and variables for recording losses
    mean_train_losses = []
    mean_val_losses = []
    best_val_loss = 100.0
    best_val_epoch = 0

    # loop through epochs
    epoch_loop = tqdm(range(config.NUM_EPOCHS), desc='Epoch')
    for epoch in epoch_loop:
        epoch_loop.set_description(f"Epoch {epoch+1}")

        model.train()   #model in training mode

        # initialize the training and validation loss for the train and val loops
        train_losses = []
        val_losses = []

        # loop through the training set
        train_loop = tqdm(trainLoader, desc='Batch')
        for step, batch in enumerate(train_loop):
            train_loop.set_description(f"Batch {step+1}")
            img = batch["image"].to(config.DEVICE)
            lbl = batch["label"].float().to(config.DEVICE)
            name = batch['name']
            
            # see the first image crop 
            #if step==0:
            #    visualizeTransformedData(img[0][0].to('cpu'),lbl[0][0].to('cpu'),60)

            # forward pass
            with torch.cuda.amp.autocast():
                predicted = model(img)
                loss = lossFunc(predicted, lbl)

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # try gradient clipping because of NaNs
            optimizer.step()

            train_losses.append(loss.item())
            
            # get binary segmentation for visualization
            predicted_prob = torch.sigmoid(predicted[0][0]) # first of the 4 crops
            predicted_label = (predicted_prob > config.THRESHOLD).astype(np.uint8)

            visualizeSegmentation(img, lbl, name, predicted_label)

            torch.cuda.empty_cache()
        
        # evaluation
        with torch.no_grad():
            model.eval()
            
            # validation loop
            for step, batch in enumerate(valLoader):
                img = batch["image"].to(config.DEVICE)
                lbl = batch["label"].float().to(config.DEVICE)
                name = batch['name']

                predicted = model(img)
                loss = lossFunc(predicted, lbl)
                val_losses.append(loss.item())
            
            # save best performing epoch
            if np.mean(val_losses) < best_val_loss:
                best_val_loss = np.mean(val_losses)
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }
                saveCheckpoint(state, 'unet_task03_liver.pth')
                print(f'Model was saved. Current best val loss {best_val_loss}')
                best_val_epoch = epoch
            else:
                print('Model was not saved.')

        # Update train and val mean losses by epoch
        print(f'Mean loss on epoch {epoch + 1}: {np.mean(train_losses)}')
        mean_train_losses.append(np.mean(train_losses))
        mean_val_losses.append(np.mean(val_losses))

    # After training plot mean losses of epochs
    plotLoss(mean_train_losses, title= "Mean Training Loss")
    plotLoss(mean_val_losses, fig_path='output/plots/validationloss.png', title='Mean Validation Loss')


        
        
    
    
    

def main():
    #parser = argparse.ArgumentParser()
    #args = parser.parse_args()

    # create loaders
    trainLoader = getLoader3d('train', 'Task03_Liver')
    valLoader = getLoader3d('val', 'Task03_Liver')
    
    #initialize model
    model = UNet3D().to(config.DEVICE)

    # initialize loss function and optimizer
    lossFunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    optimizer = AdamW(model.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY)

    # call training loop
    train(trainLoader, valLoader, model, optimizer, lossFunc)
    

if __name__ == "__main__":
    main()