# import the necessary packages
import nibabel as nib
import os
import sys
import math
import numpy as np
import pickle
from scipy.ndimage import center_of_mass

from monai.data import DataLoader, Dataset, CacheDataset

from monai.transforms import (
	Compose,
	LoadImaged,
	EnsureChannelFirstd,
	AddChanneld,
	Orientationd,
	Spacingd,
	ScaleIntensityRanged,
	CropForegroundd,
	ToTensord,
	RandCropByPosNegLabeld,
	SpatialPadd,
	Resized,
	RandRotate90d
)
import config


transforms_sam = Compose([
	LoadImaged(keys=["image", "label"], image_only=False),
	EnsureChannelFirstd(keys=["image", "label"]),
	ScaleIntensityRanged(
		keys="image",
		a_min=-175,
		a_max=250,
		b_min=0.0,
		b_max=1.0,
		clip=True,
	),
	ToTensord(keys=["image", "label"]),
])

def get_point_prompt(ground_truth_map):
	'''
	------Individual prompt for each image------
	Creates a point prompt for sam based on the centre of mass 
		from the ground truth labels 
	Returns: a tuple of the (x,y)-coordinates of the prompt
	'''
	cm = center_of_mass(ground_truth_map)
	cm = (round(cm[0]), round(cm[1]))	# round to closest int to get indices

	#print('point prompt: ', cm)
	
	return cm


def center_of_mass_from_3d(loader):
	'''
	------One prompt for all images------
	Stacks all the ground truth masks from the loader together into a 3D
	arrray to find one point describing the approximate center of mass of
	the organ defined in config.ORGAN

	Returns: the point prompt as a tuple
	'''

	print('Calculating point prompt...')

	stacked_gt_labels = np.array([])
	for i, item in enumerate(loader):
		ground_truth_mask = item['label'].squeeze().to(bool)

		if i==0:
			stacked_gt_labels = np.array([ground_truth_mask])
		else:
			stacked_gt_labels = np.vstack((stacked_gt_labels, ground_truth_mask[np.newaxis,...]))

		print(stacked_gt_labels.shape)

	cm = center_of_mass(stacked_gt_labels)
	print(cm)
	# round to closest int to get indices and
	# only keep x and y axis
	cm = (round(cm[1]), round(cm[2]))	
	print(cm)

	return cm


def averaged_center_of_mass(loader):
	'''
	Calculate just the average of the individual centers of mass
	'''
	print('Calculating the averaged center of mass for the point prompt...')
	cms_0 = []
	cms_1 = []
	for i, item in enumerate(loader):
		ground_truth_mask = item['label'].squeeze().to(bool)
		cm = center_of_mass(ground_truth_mask)
		cm_0, cm_1 = round(cm[0]), round(cm[1])
		cms_0.append(cm_0)
		cms_1.append(cm_1)
	avg_0 = sum(cms_0)/len(cms_0)
	avg_1 = sum(cms_1)/len(cms_1)

	print('avg center of mass:', avg_0, avg_1)

	return (avg_0, avg_1)



def get_loader(organ, split='train'):
	# take the first data split for SAMs maasks
	data_dicts = get_data_dicts(organ=organ, split=split)

	dataset = Dataset(data_dicts, transforms_sam)
	loader = DataLoader(dataset, batch_size=config.BATCH_SIZE_SAM, num_workers=config.NUM_WORKERS, shuffle=False)

	return loader


def get_data_dicts(organ, split='train'):
	'''
	Get the images for the given split in a data dict
	'''
	organ = organ.split('_')[1].lower()

	img_dir = os.path.join(config.DATASET_PATH_2D, f"{split}_2d_images")
	lbl_dir = os.path.join(config.DATASET_PATH_2D, f"{split}_2d_masks")

	img_fnames = []
	names = []
	for f in os.listdir(img_dir):
		task = f.split('_')[0]
		
		if task == organ and os.path.isfile(os.path.join(img_dir, f)):
			img_fnames.append(os.path.join(img_dir, f))
			names.append(f.split('.')[0])

	lbl_fnames = []
	for f in os.listdir(lbl_dir):
		task = f.split('_')[0]
		if task == organ and os.path.isfile(os.path.join(lbl_dir, f)):
			lbl_fnames.append(os.path.join(lbl_dir, f))

	data = [{"image": img_fname, "label": lbl_fname, "name": name} for img_fname, lbl_fname, name in zip(img_fnames, lbl_fnames, names)]
	
	print(f'{split} len {format(len(data))}')

	return data


def get_two_data_splits(organ, split='train'):
	'''
	Split the training images to half and half in order to 
			produce pseudo labels with SAM.
	Returns: two data dicts for training images 
	'''
	# reformat organ for 2D case
	organ = organ.split('_')[1].lower()

	img_dir = os.path.join(config.DATASET_PATH_2D, f"{split}_2d_images")
	lbl_dir = os.path.join(config.DATASET_PATH_2D, f"{split}_2d_masks")

	img_fnames = []
	names = []
	for f in os.listdir(img_dir):
		task = f.split('_')[0]
		
		if task == organ and os.path.isfile(os.path.join(img_dir, f)):
			img_fnames.append(os.path.join(img_dir, f))
			names.append(f.split('.')[0])

	lbl_fnames = []
	for f in os.listdir(lbl_dir):
		task = f.split('_')[0]
		if task == organ and os.path.isfile(os.path.join(lbl_dir, f)):
			lbl_fnames.append(os.path.join(lbl_dir, f))

	data = [{"image": img_fname, "label": lbl_fname, "name": name} for img_fname, lbl_fname, name in zip(img_fnames, lbl_fnames, names)]
	
	len_half = math.floor(len(data)/2)

	# if data len is odd, datasplit2 has 1 more img than datasplit1
	datasplit1 = data[:len_half]
	datasplit2 = data[len_half:]
	print(f'{split} split 1 len {format(len(datasplit1))}, split 2 len {format(len(datasplit2))}')

	return datasplit1, datasplit2



def get_n_worst_images(n, organ, split):
	'''
	Input:
		n - number of images to return
		organ - target organ in the format 'TaskXX_Organ', e.g. 'Task03_Liver'
		split - train, test, or val - from which data split to take the images 
		path_to_dices - the path to the .pkl file containing the ordered list
		of dice scores acquired from SAM

	Returns:
		a data dict of the n worst performing images on SAM.
	'''
	path_to_dices = os.path.join(config.SAM_OUTPUT_PATH, f'{split}_images/dices/{split}_dice_scores.pkl')
	
	with open(path_to_dices, 'rb') as f:
		dices = pickle.load(f)
	
	# take the n last elements from the list - these are the n worst dices
	n_worst = dices[-n:]

	# get data_dict for the given organ and split to search the images from
	all_cases = get_data_dicts(organ, split)

	# search for matches
	matches = []
	for (img_name, dice) in n_worst:
		match = list(filter(lambda img_dict: img_dict['name'] == img_name, all_cases))
		matches = matches + match
	
	return matches

