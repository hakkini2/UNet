from unetDataset import (
	getLoader3d,
	getLoader2d
)
from unet3d import UNet3D
from unet2d import UNet2D
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from monai.data import DataLoader
from monai.losses import DiceCELoss
from sklearn.model_selection import train_test_split
from pathlib import Path
from imutils import paths
import nibabel as nib
import shutil
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import config

from torchmetrics.functional.classification import dice
from utils.utils import calculate_dice_score


# create directories for saving test plots
if config.IMG_FORMAT == '2d':
	if config.USE_PSEUDO_LABELS:
		save_path = config.TEST_OUTPUT_PATH + config.SAM_PROMPT + '_pseudomasks/'
	else:
		save_path = config.TEST_OUTPUT_PATH + config.TRAIN_DATA + '/'
if config.IMG_FORMAT == '3d':
	save_path = config.TEST_OUTPUT_PATH + config.IMG_FORMAT + '/'
all_plots_path = save_path + 'all/'
best_plots_path = save_path + 'top_best/'
worst_plots_path = save_path + 'top_worst/'
Path(all_plots_path).mkdir(parents=True, exist_ok=True)
Path(best_plots_path).mkdir(parents=True, exist_ok=True)
Path(worst_plots_path).mkdir(parents=True, exist_ok=True)


def test(testLoader, model):
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

		if config.ORGAN in ['Task03_Liver', 'Task07_Pancreas', 'Task08_HepaticVessel']:
			lbl[lbl==2] = 1

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

		# print('dice pytorch: ', dice_pytorch)
		# print('dice utils: ', dice_utils)


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
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint_name',
                        default = 'unet_task03_liver_2d.pth',
                        help='Name of model checkpoint (.pth file)'
                        )
	args = parser.parse_args()

	# create loader
	if config.IMG_FORMAT == '3d':
		testLoader = getLoader3d('test', config.ORGAN)
	else:	#2d
		testLoader = getLoader2d('test', config.ORGAN)
	
	# get trained model from checkpoint
	if config.IMG_FORMAT == '3d':
		model = UNet3D().to(config.DEVICE)
	else:
		model = UNet2D().to(config.DEVICE)
	checkpoint_name = args.checkpoint_name
	checkpoint = torch.load(os.path.join(config.SAVED_MODEL_PATH, checkpoint_name))
	model.load_state_dict(checkpoint['model_state_dict'])

	print(f'Using model checkpoint: {checkpoint_name}')
	time.sleep(3)
	
	# run test loop for test images and get dice scores
	dices = test(testLoader, model)

	# get average dice on test set
	dices_data = [dice_item[1].item() for dice_item in dices]
	average_dice = sum(dices_data)/len(dices_data)
	print(f'\nAverage test dice score on {config.ORGAN}: {average_dice:1.4f}')

	# plot a histogram of the dice scores

	if config.TRAIN_DATA != 'all':
		train_data = config.TRAIN_DATA.split('_')
		model_specs = f'{config.N_TRAIN_SAMPLES}_{train_data[1]}'
	else:
		if config.USE_PSEUDO_LABELS:
			model_specs = f'{config.SAM_PROMPT}_pseudomasks'
		else:
			model_specs = f'{config.TRAIN_DATA}'

	plt.hist(dices_data, bins=20)
	plt.title(f'{config.ORGAN}: test dice avg: {average_dice:1.4f} ({model_specs}, {checkpoint["epoch"]} epochs)')
	plt.xlabel('Dice')
	plt.savefig(f'{save_path}/{config.ORGAN.lower()}_test_dice_histogram_{model_specs}_{config.IMG_FORMAT}.png')
	plt.close()
	

	
		

	

if __name__ == "__main__":
	main()