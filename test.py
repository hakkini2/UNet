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
import config

from torchmetrics.functional.classification import dice

def main():
	# create loaders
	testLoader = getLoader3d('test', 'Task03_Liver')
	
	# get trained model from checkpoint
	model = UNet3D().to(config.DEVICE)
	checkpoint_name = 'unet_task03_liver.pth'
	checkpoint = torch.load(os.path.join(config.SAVED_MODEL_PATH, checkpoint_name))
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	#test model for test images
	test_loop = tqdm(testLoader)
	for step, batch in enumerate(test_loop):
		test_loop.set_description(f"Iteration {step+1}")

		# get image and ground truth label
		img = batch["image"].to(config.DEVICE)
		lbl = batch["label"].float().to(config.DEVICE)
		name = batch['name']

		# get prediction
		pred = model(img)

		# evaluate with dice score
		dice_pytorch = ...

		

	

if __name__ == "__main__":
	main()