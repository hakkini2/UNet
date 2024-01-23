from unetDataset import (
    getLoader3d,
    getLoader2d
)
from unet3d import UNet3D
from unet2d import UNet2D
import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from monai.data import DataLoader
from monai.losses import DiceCELoss, DiceLoss
from sklearn.model_selection import train_test_split

from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import sys
import argparse
import math

from utils.utils import (
    plotLoss,
    visualizeTransformedData3d,
    visualizeSegmentation3d,
    visualizeSegmentation2d,
    saveCheckpoint,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(trainLoader, valLoader, model, optimizer, lossFunc, img_format):
    print('[INFO] started training the network...')

    torch.autograd.set_detect_anomaly(True) # debugging

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
            #    visualizeTransformedData3d(img[0][0].to('cpu'),lbl[0][0].to('cpu'),60)
                        
            # forward pass
            if img_format == '3d':
                with torch.cuda.amp.autocast():
                    predicted = model(img)
                    loss = lossFunc(predicted, lbl)
            else:
                predicted = model(img)
                loss = lossFunc(predicted, lbl)

            #backpropagation
            optimizer.zero_grad()
            loss.backward()

            # look for nan gradients
            for p in model.parameters():
                grad = p.grad.norm()
                if math.isnan(grad):
                    print('NAN gradient found')
                    sys.exit(0)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # try gradient clipping because of NaNs
            optimizer.step()

            train_losses.append(loss.item())
            
            # visualize 3D segmentation - to check progress while running
            if img_format == '3d' and step % config.PLOT_SAVING_INTERVAL == 0:
                # get binary segmentation for visualization
                predicted_prob = torch.sigmoid(predicted[0][0]) # first of the 4 crops
                predicted_label = (predicted_prob > config.THRESHOLD).astype(np.uint8)
                visualizeSegmentation3d(img, lbl, name, predicted_label)
            
            # visualize 2D segmentation - to check progress while running
            if img_format == '2d' and step % config.PLOT_SAVING_INTERVAL == 0:
                predicted_prob = torch.sigmoid(predicted)
                predicted_label = (predicted_prob > config.THRESHOLD).astype(np.uint8)
                visualizeSegmentation2d(img, lbl, name, predicted_label)

            torch.cuda.empty_cache()
        
        # evaluation
        with torch.no_grad():
            model.eval()
            print('Validating model...')

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
                saveCheckpoint(state)
                print(f'Model was saved. Current best val loss {best_val_loss}')
                best_val_epoch = epoch
            else:
                print(f'Model was not saved. Val loss was {np.mean(val_losses)}, but current best val loss is {best_val_loss}')

        # Update train and val mean losses by epoch
        print(f'Mean train loss on epoch {epoch + 1}: {np.mean(train_losses)}')
        mean_train_losses.append(np.mean(train_losses))
        mean_val_losses.append(np.mean(val_losses))

    # After training, plot mean losses of epochs
    if config.IMG_FORMAT == '3d':
        plotLoss(mean_train_losses, fig_path=f'{config.SAVED_PLOTS_PATH}loss_{config.ORGAN}_{img_format}.png',
                title= f'{config.ORGAN}: {img_format.upper()} Mean Training and Validation Loss')

    
    if config.IMG_FORMAT == '2d':
        if config.TRAIN_DATA != 'all':
            train_data = config.TRAIN_DATA.split('_')
            plot_text = f'{config.N_TRAIN_SAMPLES} {train_data[1]} train images'
            fname_text = f'{config.N_TRAIN_SAMPLES}_{train_data[1]}'
        else:
            plot_text = f'{config.TRAIN_DATA} train images'
            fname_text = f'{config.TRAIN_DATA}'

        plotLoss(mean_train_losses, mean_val_losses, fig_path=f'{config.SAVED_PLOTS_PATH}loss_{config.ORGAN}_{fname_text}_{img_format}.png',
                title= f'{config.ORGAN}: {img_format.upper()} Mean Training and Validation Loss, {plot_text}')
    
    
    

def main():
    image_format = config.IMG_FORMAT  # 3d or 2d

    if image_format == '3d':
        # create loaders
        trainLoader = getLoader3d('train', config.ORGAN)
        valLoader = getLoader3d('val', config.ORGAN)
        
        #initialize model
        model = UNet3D().to(config.DEVICE)

        # initialize loss function and optimizer
        lossFunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        optimizer = AdamW(model.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY)

        # call training loop
        train(trainLoader, valLoader, model, optimizer, lossFunc, image_format)
    

    if image_format == '2d':
        # create loaders
        trainLoader = getLoader2d('train', config.ORGAN)
        valLoader = getLoader2d('val', config.ORGAN)
        
        #initialize model
        model = UNet2D().to(config.DEVICE)

        # initialize loss function and optimizer
        #lossFunc = DiceLoss(sigmoid=True, reduction='mean')
        lossFunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        optimizer = AdamW(model.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY)

        # call training loop
        train(trainLoader, valLoader, model, optimizer, lossFunc, image_format)

if __name__ == "__main__":
    main()