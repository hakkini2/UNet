# import the necessary packages
import nibabel as nib
import config
import os

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
	SpatialPadd
)


train_transforms =  Compose([
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
			# extracts intensity range [-57, 164] and scales to [0, 1].
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
			CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=(256,256,256), mode='constant'),
			
			# randomly crop patch samples from big image based on pos / neg ratio
			#RandCropByPosNegLabeld(
			#	keys=["image", "label"],
			#	label_key="label",
			#	spatial_size=(96, 96, 96),
			#	pos=2,
			#	neg=1,
			#	num_samples=4,
			#	image_key="image",
			#	image_threshold=0,
        	#),
		])

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
    ]
)



def getLoader(phase, organ):
	'''
	phase (String): train, test or val
	organ(String): e.g. 'Task03_Liver'
	'''
	if phase == 'train':
		data_dicts_train = getTrainPaths(organ)
		train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
		train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS)

		return train_loader

	if phase == 'val':
		data_dicts_val = getValPaths(organ)
		val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
		val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS)

		return val_loader
	
	if phase == 'test':
		data_dicts_test = getTestPaths(organ)
		test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
		test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS)

		return test_loader



def getTrainPaths(organ = 'Task03_Liver'):
	train_img = []
	train_lbl = []
	train_name = []

	for line in open(config.DATA_TXT_PATH_TRAIN):
		task = line.strip().split()[0].split('/')[1]    # e.g. 'Task03_Liver'
		if task == organ:
			name = line.strip().split()[1].split('.')[0]
			train_name.append(name)
			train_img.append(config.DATASET_PATH + line.strip().split()[0])
			train_lbl.append(config.DATASET_PATH + line.strip().split()[1])

	data_dicts_train = [{'image': image, 'label': label, 'name': name}
				for image, label, name in zip(train_img, train_lbl, train_name)]
	
	print('train len {}'.format(len(data_dicts_train)))
	#print('train len: {}'.format(len(train_img)))

	#return train_img, train_lbl
	return data_dicts_train


def getValPaths(organ = 'Task03_Liver'):
	val_img = []
	val_lbl = []
	val_name = []

	for line in open(config.DATA_TXT_PATH_VAL):
		task = line.strip().split()[0].split('/')[1]    # e.g. 'Task03_Liver'
		if task == organ:
			name = line.strip().split()[1].split('.')[0]
			val_name.append(name)
			val_img.append(config.DATASET_PATH + line.strip().split()[0])
			val_lbl.append(config.DATASET_PATH + line.strip().split()[1])

	data_dicts_val = [{'image': image, 'label': label, 'name': name}
				for image, label, name in zip(val_img, val_lbl, val_name)]
	
	print('val len {}'.format(len(data_dicts_val)))
	#print('val len: {}'.format(len(val_img)))

	#return val_img, val_lbl
	return data_dicts_val


def getTestPaths(organ = 'Task03_Liver'):
	test_img = []
	test_lbl = []
	test_name = []

	for line in open(config.DATA_TXT_PATH_TEST):
		task = line.strip().split()[0].split('/')[1]    # e.g. 'Task03_Liver'
		if task == organ:
			name = line.strip().split()[1].split('.')[0]
			test_name.append(name)
			test_img.append(config.DATASET_PATH + line.strip().split()[0])
			test_lbl.append(config.DATASET_PATH + line.strip().split()[1])

	data_dicts_test = [{'image': image, 'label': label, 'name': name}
	            for image, label, name in zip(test_img, test_lbl, test_name)]
	
	print('test len {}'.format(len(data_dicts_test)))
	#print('test len: {}'.format(len(test_img)))

	#return test_img, test_lbl
	return data_dicts_test
