import os
import SimpleITK as sitk
import numpy as np
import argparse



def getDataDicts(args, img_type):
    list_img = []
    list_lbl = []
    list_name = []

    for line in open(args.data_txt_path + '_' + img_type + '.txt'):
        name = line.strip().split()[1].split('.')[0]
        list_img.append(args.data_dir + line.strip().split()[0])
        list_lbl.append(args.data_dir + line.strip().split()[1])
        list_name.append(name)
    data_dicts = [{'image': image, 'label': label, 'name': name}
                for image, label, name in zip(list_img, list_lbl, list_name)]
    
    print(img_type, 'len {}'.format(len(data_dicts)))

    return data_dicts



def make2dSlices(args, img_type, data_dicts):
    '''
    Creates2d images and 2d masks under /content
    Run with:
        make2dSlices(args, 'train', data_dicts_train)
        make2dSlices(args, 'val', data_dicts_val)
        make2dSlices(args, 'test', data_dicts_test)

        where data_dicts_<type> is obtained from getDataDicts(args, <type>)
    '''

    # create directories for 2d slices
    dir_paths = {}

    for data_type in ['2d_images', '2d_masks']:
        # Construct the directory path
        dir_path = os.path.join(args.base_dir, f'{img_type}_{data_type}')
        dir_paths[f'{img_type}_{data_type}'] = dir_path
        # Create the directory
        os.makedirs(dir_path, exist_ok=True)

    # make slices
    for idx, item in enumerate(data_dicts):
        # get paths to image and label
        img_path = item['image']
        lbl_path = item['label']
        name_path = item['name']

        # load image and label
        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(lbl_path)
        print('processing patient', idx, img.GetSize(), mask.GetSize())

        # Get the mask data as numpy array
        mask_data = sitk.GetArrayFromImage(mask)

        # select correct directories
        img_dir = dir_paths[f'{img_type}_2d_images']
        mask_dir = dir_paths[f'{img_type}_2d_masks']
        print('mask data shape',np.shape(mask_data))
        # iterate over the axial slices
        for i in range(img.GetSize()[2]):
            # If the mask slice is not empty, save the image and mask slices
            if np.any(mask_data[i, :, :]):
                # Prepare the new ITK images
                img_slice = img[:, :, i]
                mask_slice = mask[:, :, i]
                
                # Define the output paths
                img_slice_path = os.path.join(img_dir, f"{os.path.basename(img_path).replace('.nii.gz', '')}_{i}.nii.gz")
                mask_slice_path = os.path.join(mask_dir, f"{os.path.basename(lbl_path).replace('.nii.gz', '')}_{i}.nii.gz")

                # Save the slices as NIfTI files
                sitk.WriteImage(img_slice, img_slice_path)
                sitk.WriteImage(mask_slice, mask_slice_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        default = './content',
                        help='The path where to save the 2d slices'
                        )
    parser.add_argument('--data_dir',
                        default='/l/ComputerVision/CLIP-and-SwinUNETR/Swin-UNETR-with-MSD/data/',
                        help = 'directory where to find MSD data'
                        )
    parser.add_argument('--data_txt_path',
                        default = './dataset/dataset_list/PAOT_10_inner',
                        help = 'Beginning pattern of path to txt file describing the train/val/test spits'
                        ) 
    args = parser.parse_args()

    # get data dicts of train, val and test splits
    data_dicts_train = getDataDicts(args, 'train')
    data_dicts_val = getDataDicts(args, 'val')
    data_dicts_test = getDataDicts(args, 'test')

    # make the 2d slices and save them to --base_dir
    make2dSlices(args, 'train', data_dicts_train)
    make2dSlices(args, 'val', data_dicts_val)
    make2dSlices(args, 'test', data_dicts_test)



if __name__ == "__main__":
    main()
