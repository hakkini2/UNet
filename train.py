from dataset.dataset import (
    SegmentationDataset,
    getTrainPaths,
    getValPaths,
    getTestPaths
)
from model import UNet
import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os



def main():
    # get training, validation and test image and mask paths
    # we only use Liver for now
    train_paths = getTrainPaths(organ = 'Task03_Liver')
    val_paths = getValPaths(organ = 'Task03_Liver')
    test_paths = getTestPaths(organ = 'Task03_Liver')

    # create the train and test datasets
    trainDS = SegmentationDataset(train_paths)
    testDS = SegmentationDataset(test_paths)
    
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # create the training and test data loaders
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # check if the dataloader needs to be changed from pytorch to monai
    # because of the 3D images
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count())
    
    testLoader = DataLoader(testDS, shuffle=False,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count())
    


    


if __name__ == "__main__":
    main()