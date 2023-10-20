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
import torch
import time
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(trainLoader, model, optimizer, lossFunc):
    print('[INFO] started training the network...')

    # loop through epochs
    for epoch in tqdm(range(config.NUM_EPOCHS), desc='Epoch'):
        model.train()   #model in training mode

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # loop through the training set
        train_loop = tqdm(trainLoader, desc='Batch')
        for step, batch in enumerate(train_loop):
            img = batch["image"].to(config.DEVICE)
            lbl = batch["label"].float().to(config.DEVICE)
            name = batch['name']

            # see the fist image 
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

            totalTrainLoss += loss

            #update tqdm loop
            train_loop.set_postfix(loss=loss.item())

            torch.cuda.empty_cache()



        
        
        
    
    

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
    lossFunc = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = AdamW(model.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY)

    # call training loop
    train(trainLoader, model, optimizer, lossFunc)

    #x = torch.randn((3, 1, 256, 256, 256))
    #x = x.to(config.DEVICE)
    #preds = model(x)
    #print(preds.shape)
    #print(x.shape)
    

if __name__ == "__main__":
    main()