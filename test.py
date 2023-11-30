from dataset import (
	getLoader3d,
	getLoader2d
)
from model.unet3d import UNet3D
from model.unet2d import UNet2D
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from monai.data import DataLoader
from monai.losses import DiceCELoss
from sklearn.model_selection import train_test_split
from pathlib import Path
from imutils import paths
import nibabel as nib
import shutil
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
	# create directories for saving test plots
	all_plots_path = config.TEST_OUTPUT_PATH + 'all/'
	best_plots_path = config.TEST_OUTPUT_PATH + 'top_best/'
	worst_plots_path = config.TEST_OUTPUT_PATH + 'top_worst/'
	Path(all_plots_path).mkdir(parents=True, exist_ok=True)
	Path(best_plots_path).mkdir(parents=True, exist_ok=True)
	Path(worst_plots_path).mkdir(parents=True, exist_ok=True)
	
	# set model to evaluation mode
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
		print(img.shape)
		# get prediction
		pred = model(img)
		
		# get binary segmentation
		predicted_prob = torch.sigmoid(pred)
		predicted_label = (predicted_prob > config.THRESHOLD).astype(np.uint8)
		predicted_label = torch.Tensor(predicted_label)

		# squeeze extra dimension and convert masks to bool 
		img = img.squeeze()
		lbl = lbl.squeeze().to(bool)
		predicted_label = predicted_label.squeeze().to(bool)

		# evaluate with dice score
		dice_pytorch = dice(predicted_label.cpu(), lbl.cpu(), ignore_index=0)
		dice_utils, _, _ = calculate_dice_score(predicted_label.cpu(), lbl.cpu())

		print('dice pytorch: ', dice_pytorch)
		print('dice utils: ', dice_utils)


		# check that different dice scores match
		if not np.isclose(dice_pytorch, dice_utils.item()):
			print("DIFFERENT DICES \n")
			print(f"i: {step}, name: {name[0]}")
			break

		dices.append((name[0], dice_pytorch))

		#visualize and save plots
		with torch.no_grad():
			plt.figure(figsize=(12,4))
			plt.suptitle(f'{name[0]}, dice: {dice_utils.item():1.4f}', fontsize=14)
			plt.subplot(1,3,1)
			plt.imshow(img.to('cpu'), cmap='gray')
			plt.axis('off')
			plt.subplot(1,3,2)
			plt.imshow(lbl.to('cpu'), cmap='copper')
			plt.axis('off')
			plt.subplot(1,3,3)
			plt.imshow(predicted_label, cmap='copper')
			plt.axis('off')
			plt.tight_layout()
			plt.savefig(f'{all_plots_path}{name[0]}_{config.IMG_FORMAT}.png')
			plt.close()
	
	# sort the dices by the dice score (highest first)
	dices.sort(key=lambda x: x[1].item(), reverse=True)

	# top config.N_TEST_SAMPLES best and worst segmentation results
	top_best, top_worst = dices[: config.N_TEST_SAMPLES], dices[-config.N_TEST_SAMPLES :]

	# copy the corresponding samples from all/ to top_best/ and top_worst/
	for fname, _ in top_best:
		shutil.copy(f'{all_plots_path}{fname}_{config.IMG_FORMAT}.png', f'{best_plots_path}{fname}_{config.IMG_FORMAT}.png')
	for fname, _ in top_worst:
		shutil.copy(f'{all_plots_path}{fname}_{config.IMG_FORMAT}.png', f'{worst_plots_path}{fname}_{config.IMG_FORMAT}.png')

	return dices


def main():
	# get image format the model was trained on (2d / 3d)
	img_format = config.IMG_FORMAT

	# create loader
	if img_format == '3d':
		testLoader = getLoader3d('test', config.ORGAN)
	else:	#2d
		testLoader = getLoader2d('test', config.ORGAN)
	
	# get trained model from checkpoint
	if img_format == '3d':
		model = UNet3D().to(config.DEVICE)
	else:
		model = UNet2D().to(config.DEVICE)
	checkpoint_name = f'unet_{config.ORGAN.lower()}_{img_format}.pth'
	checkpoint = torch.load(os.path.join(config.SAVED_MODEL_PATH, checkpoint_name))
	model.load_state_dict(checkpoint['model_state_dict'])

	# run test loop for test images and get dice scores
	dices = test(testLoader, model, img_format)

	# get average dice on test set
	dices_data = [dice_item[1].item() for dice_item in dices]
	average_dice = sum(dices_data)/len(dices_data)
	print(f'\nAverage test dice score on {config.ORGAN}: {average_dice:1.4f}')

	# plot a histogram of the dice scores
	plt.hist(dices_data, bins=20)
	plt.title(f'{config.ORGAN} test dice histogram, {config.IMG_FORMAT}, avg: {average_dice:1.4f}')
	plt.xlabel('Dice')
	plt.savefig(f'{config.TEST_OUTPUT_PATH}{config.ORGAN.lower()}_test_dice_histogram_{config.IMG_FORMAT}.png')
	plt.close()
	

	
		

	

if __name__ == "__main__":
	main()