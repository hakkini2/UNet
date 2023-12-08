import torch
import os

# Image format: are the input images 3D volumes or 2D slices
IMG_FORMAT = '2d'   # '2d' or '3d'

PLOT_SAVING_INTERVAL = 200  # Save model predictions to a plot by every Nth batch in training
N_TEST_SAMPLES = 10   # How many top/worst test plots are saved

#organ to train/test on
# NOTE: THE FORMAT TaskXX_Organ
ORGAN = 'Task03_Liver' #'Task03_Liver'

# path to dataset
DATASET_PATH_3D = '/l/ComputerVision/CLIP-and-SwinUNETR/Swin-UNETR-with-MSD/data/'
DATASET_PATH_2D = './content/'

# paths to txt files describing the 3D dataset splits
DATA_TXT_PATH_TRAIN = './dataset/dataset_list/PAOT_10_inner_train.txt'
DATA_TXT_PATH_VAL = './dataset/dataset_list/PAOT_10_inner_val.txt'
DATA_TXT_PATH_TEST = './dataset/dataset_list/PAOT_10_inner_test.txt'

# path to saved SAM model checkpoint
SAM_CHECKPOINT_PATH = './content/model_checkpoints/sam_vit_h_4b8939.pth'

SAM_OUTPUT_PATH = 'output/sam/'

# define the path to the base output directory
BASE_OUTPUT = 'output/unet'
TEST_OUTPUT_PATH = 'output/unet/plots/test_plots/'
SAVED_PLOTS_PATH = 'output/unet/plots/'
SAVED_MODEL_PATH = 'output/unet/pretrained/'

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
INIT_LR = 0.0005
NUM_EPOCHS = 16
if IMG_FORMAT == '2d':
    BATCH_SIZE = 2
else:
    BATCH_SIZE = 1  # issue with different sized tensors inside a batch
BATCH_SIZE_SAM = 1 # batch size for SAM
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0 # os.cpu_count()

# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_DEPTH = 256 # ???

# define threshold to filter weak predictions
THRESHOLD = 0.5

