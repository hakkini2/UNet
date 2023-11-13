# import the necessary packages
import nibabel as nib
import os
import sys
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

train_transforms_3d =  Compose([
	# loads the spleen CT images and labels from NIfTI format files
	LoadImaged(keys=['image', 'label']),
	#ensures the original data to construct "channel first" shape
	EnsureChannelFirstd(keys=["image", "label"]),
	# unifies the data orientation based on the affine matrix
	Orientationd(keys=["image", "label"], axcodes="RAS"),
	# adjusts the spacing by pixdim based on the affine matrix
	Spacingd(
		keys=["image","label"],
		pixdim=(1.5, 1.5, 1.5),
		mode=("bilinear", "nearest"),
	), 
	# extracts intensity range and scales to [0, 1].
	ScaleIntensityRanged(
		keys=["image"],
		a_min=-175,
		a_max=250,
		b_min=0.0,
		b_max=1.0,
		clip=True,
	),
	CropForegroundd(keys=["image", "label"], source_key="image"),
	
	SpatialPadd(keys=["image", "label"], spatial_size=(96,96,96), mode='constant'),
	
	# randomly crop patch samples from big image based on pos / neg ratio
	# creates four patches from one image
	RandCropByPosNegLabeld(
		keys=["image", "label"],
		label_key="label",
		spatial_size=(96, 96, 96),
		pos=2,
		neg=1,
		num_samples=4,
		image_key="image",
		image_threshold=0,
	),
	ToTensord(keys=["image", "label"]),
])

train_transforms_2d = Compose([
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
	CropForegroundd(keys=["image", "label"], source_key="image"),
	SpatialPadd(keys=["image", "label"], spatial_size=(256,256), mode='constant'),
	RandCropByPosNegLabeld(
		keys=["image", "label"],
		label_key="label",
		spatial_size=(256, 256),
		pos=2,
		neg=1,
		num_samples=4,
		image_key="image",
		image_threshold=0,
	),
	RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
	ToTensord(keys=["image", "label"]),
])


val_transforms_3d = Compose([
	LoadImaged(keys=["image", "label"]),
	EnsureChannelFirstd(keys=["image", "label"]),
	Orientationd(keys=["image", "label"], axcodes="RAS"),
	Spacingd(
		keys=["image", "label"],
		pixdim=(1.5, 1.5, 1.5),
		mode=("bilinear", "nearest")
	),
	ScaleIntensityRanged(
		keys=["image"],
		a_min=-175,
		a_max=250,
		b_min=0.0,
		b_max=1.0,
		clip=True,
	),
	CropForegroundd(keys=["image", "label"], source_key="image"),
	ToTensord(keys=["image", "label"]),
])

val_transforms_2d = Compose([
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


def getLoader3d(split, organ):
	'''
	split (String): train, test or val
	organ (String): e.g. 'Task03_Liver'
	'''
	if split == 'train':
		data_dicts_train = getTrainPaths3d(organ)
		train_dataset = Dataset(data=data_dicts_train, transform=train_transforms_3d)
		train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
							num_workers=config.NUM_WORKERS)

		return train_loader

	if split == 'val':
		data_dicts_val = getValPaths3d(organ)
		val_dataset = Dataset(data=data_dicts_val, transform=val_transforms_3d)
		val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
						  num_workers=config.NUM_WORKERS)

		return val_loader
	
	if split == 'test':
		data_dicts_test = getTestPaths3d(organ)
		test_dataset = Dataset(data=data_dicts_test, transform=val_transforms_3d)
		test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
						   num_workers=config.NUM_WORKERS)

		return test_loader


def getLoader2d(split, organ):
	data_dicts = getDataPaths2d(split=split, organ=organ)

	if split=='train':
		dataset = Dataset(data=data_dicts, transform=train_transforms_2d)
	else:
		dataset = Dataset(data=data_dicts, transform=val_transforms_2d)

	loader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True,
					num_workers=config.NUM_WORKERS)

	return loader

# 3D data paths

def getTrainPaths3d(organ = 'Task03_Liver'):
	train_img = []
	train_lbl = []
	train_name = []

	for line in open(config.DATA_TXT_PATH_TRAIN):
		task = line.strip().split()[0].split('/')[1]    # e.g. 'Task03_Liver'
		if task == organ:
			name = line.strip().split()[1].split('.')[0].split('/')[-1]
			train_name.append(name)
			train_img.append(config.DATASET_PATH_3D + line.strip().split()[0])
			train_lbl.append(config.DATASET_PATH_3D + line.strip().split()[1])

	data_dicts_train = [{'image': image, 'label': label, 'name': name}
				for image, label, name in zip(train_img, train_lbl, train_name)]
	
	print('train len {}'.format(len(data_dicts_train)))
	#print('train len: {}'.format(len(train_img)))

	#return train_img, train_lbl
	return data_dicts_train


def getValPaths3d(organ = 'Task03_Liver'):
	val_img = []
	val_lbl = []
	val_name = []

	for line in open(config.DATA_TXT_PATH_VAL):
		task = line.strip().split()[0].split('/')[1]    # e.g. 'Task03_Liver'
		if task == organ:
			name = line.strip().split()[1].split('.')[0].split('/')[-1]
			val_name.append(name)
			val_img.append(config.DATASET_PATH_3D + line.strip().split()[0])
			val_lbl.append(config.DATASET_PATH_3D + line.strip().split()[1])

	data_dicts_val = [{'image': image, 'label': label, 'name': name}
				for image, label, name in zip(val_img, val_lbl, val_name)]
	
	print('val len {}'.format(len(data_dicts_val)))
	#print('val len: {}'.format(len(val_img)))

	#return val_img, val_lbl
	return data_dicts_val


def getTestPaths3d(organ = 'Task03_Liver'):
	test_img = []
	test_lbl = []
	test_name = []

	for line in open(config.DATA_TXT_PATH_TEST):
		task = line.strip().split()[0].split('/')[1]    # e.g. 'Task03_Liver'
		if task == organ:
			name = line.strip().split()[1].split('.')[0].split('/')[-1]
			test_name.append(name)
			test_img.append(config.DATASET_PATH_3D + line.strip().split()[0])
			test_lbl.append(config.DATASET_PATH_3D + line.strip().split()[1])

	data_dicts_test = [{'image': image, 'label': label, 'name': name}
	            for image, label, name in zip(test_img, test_lbl, test_name)]
	
	print('test len {}'.format(len(data_dicts_test)))
	#print('test len: {}'.format(len(test_img)))

	#return test_img, test_lbl
	return data_dicts_test



# 2D data paths
def getDataPaths2d(split, organ = 'Task03_Liver'):
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
	
	print(f'{split} len {format(len(data))}')

	return data
