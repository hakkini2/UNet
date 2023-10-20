import torch
import os

# path to dataset
DATASET_PATH = '/l/ComputerVision/CLIP-and-SwinUNETR/Swin-UNETR-with-MSD/data/'
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, '10_Decathlon/Task03_Liver/imagesTr/') # redundant?
MASK_DATASET_PATH = os.path.join(DATASET_PATH, '10_Decathlon/Task03_Liver/labelsTr/') #redundant?

# paths to txt files describing the dataset splits
DATA_TXT_PATH_TRAIN = './dataset/dataset_list/PAOT_10_inner_train.txt'
DATA_TXT_PATH_VAL = './dataset/dataset_list/PAOT_10_inner_val.txt'
DATA_TXT_PATH_TEST = './dataset/dataset_list/PAOT_10_inner_test.txt'

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1 # 2?
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 2
BATCH_SIZE = 1
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0 # os.cpu_count()

# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_DEPTH = 256 # ???

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = 'output'

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_task03_liver.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])