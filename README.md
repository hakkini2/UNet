# UNet

### about data formatting 

MSD data is in folder **DATASET_PATH**, defined in *config.py*. The data in this folder follows the format given in the files *dataset/dataset_list/PAOT_10_inner_test.txt*, *dataset/dataset_list/PAOT_10_inner_train.txt* and *dataset/dataset_list/PAOT_10_inner_val.txt*.


## Using 2D data

If using 2D data, in *config.py*, the variable **IMG_FORMAT** must be set to *'2d'*. 2D slices of the original 3D images must be created with the script *create-2d-slices.py*, which creates the train, test and val splits of the 2D images and masks under *content/*. After this, training and testing can be run for the chosen organ, which is defined in *config.py* in variable **ORGAN**. The organ string must follow the pattern *TaskXX_Organ*, for example *Task03_Liver*.