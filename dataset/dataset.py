# import the necessary packages
import nibabel as nib
import config
from torch.utils.data import Dataset

from monai.transforms import (
	Compose,
	LoadImaged,
	EnsureChannelFirstd
)

class SegmentationDataset(Dataset):
	def __init__(self, data_dicts):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.data_dicts = data_dicts
		self.transforms = transforms =  Compose([
			LoadImaged(keys=['image', 'label']),
			...
		])
		
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.data_dicts)
	
	def __getitem__(self, idx):
		# grab the image path from the current index
		# apply the transformations to both image and its mask
		image = self.transforms(self.data_dicts[idx]['image'])
		mask = self.transforms(self.data_dicts[idx]['label'])
			
		# return a tuple of the image and its mask
		return (image, mask)




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
	
	print('train len {}'.format(len(data_dicts_val)))
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
	
	print('train len {}'.format(len(data_dicts_test)))
	#print('test len: {}'.format(len(test_img)))

	#return test_img, test_lbl
	return data_dicts_test
