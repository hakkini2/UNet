# SAM and UNet

### about data formatting 

MSD data is in folder **DATASET_PATH**, defined in *config.py*. The data in this folder follows the format given in the files *dataset/dataset_list/PAOT_10_inner_test.txt*, *dataset/dataset_list/PAOT_10_inner_train.txt* and *dataset/dataset_list/PAOT_10_inner_val.txt*.


## Using 2D data

If using 2D data, in *config.py*, the variable **IMG_FORMAT** must be set to *'2d'*. 2D slices of the original 3D images must be created with the script *create-2d-slices.py*, which creates the train, test and val splits of the 2D images and masks under *content/*. After this, training and testing can be run for the chosen organ, which is defined in *config.py* in variable **ORGAN**. The organ string must follow the pattern *TaskXX_Organ*, for example *Task03_Liver*.

<!-- ### N worst, N random or all train images (2D)

The training data to be used is defined in *config.py* with variables **TRAIN_DATA** and **N_TRAIN_SAMPLES**. The value for **TRAIN_DATA** is chosen from the list **TRAIN_DATA_LIST = ['all', 'n_random', 'n_worst']**, which contains three options: using all training images, using N random images or N worst performing training images (ranked by SAM inference with point prompt). The variable **N_TRAIN_SAMPLES** defines N; how many training images are included in N worst or N random.

Training is then simply done by ```python3 unetTrain.py``` -->

### Using SAM's pseudo labels for training UNet

To save SAM's predictions as pseudo masks, the file *sam-pseudo-labels.py* must be run for the train split of the appropriate organ (organ defined in *config.py* with **ORGAN**) and the chosen prompt type (specified in *config.py* with **SAM_PROMPT**, can be 'point' or 'box'). The created pseudo masks are the nsaved  under *content/train_2d_box_pseudomasks/*, or  under *content/train_2d_point_pseudomasks/*, depending on the prompt type.

Then, with variable **USE_PSEUDO_LABELS** set as **True**, along with the desired prompt type specified in **SAM_PROMPT**, both in *config.py*, UNet can be trained using pseudo labels instead of real ground truth data by running *unetTrain.py* for the appropriate organ.

#### Testing

When testing pretrained models (saved in *output/unet/pretrained*), the model to be trained needs to be specified with an argument ```--checkpoint_name```, e.g.:

```python3 unetTest.py --checkpoint_name 'unet_task03_liver_box_pseudolabels_2D.pth'```