from dataset import (
	getLoader3d,
	getLoader2d
)
from unet3d import UNet3D
from unet2d import UNet2D
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
from utils.utils import calculate_dice_score


def test(testLoader, model, img_format):
	model.eval()

	#test model for test images
	dices = []
	test_loop = tqdm(testLoader)
	for step, batch in enumerate(test_loop):
		test_loop.set_description(f"Iteration {step+1}")

		# get image and ground truth label
		img = batch['image'].to(config.DEVICE)
		lbl = batch['label'].float().to(config.DEVICE)
		name = batch['name']

		# get prediction
		pred = model(img)

		# evaluate with dice score
		dice_pytorch = dice(pred.cpu(), lbl.cpu(), ignore_index=0)
		dice_utils = calculate_dice_score(pred.cpu(), lbl.cpu())


def main():
	# get image format the model was trained on (2d / 3d)
	img_format = config.IMG_FORMAT

	# create loader
	if img_format == '3d':
		testLoader = getLoader3d('test', 'Task03_Liver')
	else:	#2d
		testLoader = getLoader2d('test', 'Task03_Liver')
	
	# get trained model from checkpoint
	if img_format == '3d':
		model = UNet3D().to(config.DEVICE)
	else:
		model = UNet2D().to(config.DEVICE)
	checkpoint_name = f'unet_task03_liver_{img_format}.pth'
	checkpoint = torch.load(os.path.join(config.SAVED_MODEL_PATH, checkpoint_name))
	model.load_state_dict(checkpoint['model_state_dict'])

	# run test loop for test images
	test(testLoader, model, img_format)
	

	
		

	

if __name__ == "__main__":
	main()