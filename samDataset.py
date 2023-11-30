# import the necessary packages
import nibabel as nib
import os
import sys
import math
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
	Creates a point prompt for sam based on the centre of mass 
		from the ground truth labels 
	Returns: a tuple of the (x,y)-coordinates of the prompt
	'''
	cm = center_of_mass(ground_truth_map)
	cm = (round(cm[0]), round(cm[1]))	# round to closest int to get indices

	print('point prompt: ', cm)
	
	return cm



def get_loader(organ, split='train'):
	# take the first data split for SAMs maasks
	data_dicts, _ = get_two_data_splits(organ=organ, split=split)

	dataset = Dataset(data_dicts, transforms_sam)
	loader = DataLoader(dataset, batch_size=config.BATCH_SIZE_SAM, num_workers=config.NUM_WORKERS, shuffle=False)

	return loader


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


loader = get_loader(organ = 'Task03_Liver', split= 'train')

