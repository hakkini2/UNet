import os
from pathlib import Path
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from monai.data import DataLoader, Dataset
from monai.transforms import (
	Compose,
	LoadImaged,
	EnsureChannelFirstd,
	ScaleIntensityRanged,
	ToTensord,
)
from torchmetrics.functional.classification import dice
from segment_anything import SamPredictor, sam_model_registry
from unet2d import UNet2D
from utils.utils import (
    calculate_dice_score,
    show_mask,
    show_points,
    show_box,
    normalize8
)
from utils.prompts import (
	get_point_prompt_prediction,
	get_box_prompt_prediction,
	get_box_with_points_prediction,
	get_box_and_point_prompt_prediction
)
import config




val_transforms = Compose([
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



split = 'test'

output_base_path = './output/result_plots/'
Path(output_base_path).mkdir(parents=True, exist_ok=True)


def get_loader(image_list):
    data_dicts = get_images_dict(image_list)
    dataset = Dataset(data_dicts, val_transforms)
    loader = DataLoader(dataset, batch_size=1, num_workers=config.NUM_WORKERS, shuffle=False)   
    return loader


def get_images_dict(image_list):
	'''
	image_list: list of images to include
	'''
	img_dir = os.path.join(config.DATASET_PATH_2D, f"{split}_2d_images")
	lbl_dir = os.path.join(config.DATASET_PATH_2D, f"{split}_2d_masks")
	
	img_fnames = []
	lbl_fnames = []
	names = []

	for img_name in image_list:
		img_fname = os.path.join(img_dir, img_name)
		img_fnames.append(img_fname)
		lbl_fname = os.path.join(lbl_dir, img_name)
		lbl_fnames.append(lbl_fname)
		name = img_name.split('.')[0]
		names.append(name)

	data = [{"image": img_fname, "label": lbl_fname, "name": name} for img_fname, lbl_fname, name in zip(img_fnames, lbl_fnames, names)]
		
	print(f'{split} data dict of {format(len(data))} images created')

	return data





def plot_images(prompt, loader, predictor, unet_list, unet_pseudo_labels_list):

    print(f'using prompt:{prompt}')

    plt.figure(figsize=(24, 16))

    with torch.no_grad():
        for step, item in enumerate(loader):
            image_orig = item['image'].squeeze()
            ground_truth_mask = item['label'].squeeze().to(bool)
            name = item['name']
            organ = name[0].split('_')[0]

            print(f'plotting for image {name}')

            # SAM
            print('Getting SAMs prediction')
            # convert to rbg uint8 image
            color_img = cv2.cvtColor(image_orig.numpy(), cv2.COLOR_GRAY2RGB)
            color_img = normalize8(color_img)

            # process image to get image embedding
            predictor.set_image(color_img)

            # Predict using point prompts
            if prompt == 'point' or prompt == 'naive_point' or prompt == 'furthest_from_edges_point':
                prediction_sam, input_points, input_labels = get_point_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    point_type=prompt
                )

            # Predict using bounding box
            if prompt == 'box':
                prediction_sam, input_boxes = get_box_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor
                )
            
            # Predict using one bounding box and cluster points
            if prompt == 'one_box_with_points':
                prediction_sam, input_boxes, input_points, input_labels = get_box_with_points_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor
                )
            
            # Predict using a box and a point per cluster
            if prompt == 'box_and_point':
                prediction_sam, input_boxes, input_points, input_labels = get_box_and_point_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    background_point=False
                )
            
            #Predict using a box and a background point per cluster
            if prompt == 'box_and_background_point':
                prediction_sam, input_boxes, input_points, input_labels = get_box_and_point_prompt_prediction(
                    ground_truth_mask=ground_truth_mask,
                    predictor=predictor,
                    background_point=True
                )

            dice_sam = dice(torch.Tensor(prediction_sam).cpu(), ground_truth_mask.cpu(), ignore_index=0)


            # UNET and UNET WITH PSEUDOLABELS
            print('Geting UNet based predictions')

            # get input data in right format (no squeezing)
            img = item['image']
            lbl = item['label'].float()

            # checks if config.ORGAN in ['Task03_Liver', 'Task07_Pancreas', 'Task08_HepaticVessel']:
            if step in [0,2,3]:
                lbl[lbl==2] = 1

            # get predictions
            unet = unet_list[step]
            pred = unet(img)

            unet_pseudo = unet_pseudo_labels_list[step]
            pred_pseudo = unet_pseudo(img)
            
            # get binary segmentation
            predicted_prob = torch.sigmoid(pred)
            prediction_unet = (predicted_prob > config.THRESHOLD).astype(np.uint8)
            prediction_unet = torch.Tensor(prediction_unet)
            
            predicted_prob_pseudo = torch.sigmoid(pred_pseudo)
            prediction_unet_pseudo = (predicted_prob_pseudo > config.THRESHOLD).astype(np.uint8)
            prediction_unet_pseudo = torch.Tensor(prediction_unet_pseudo)

            img = img.squeeze()
            lbl = lbl.squeeze().to(bool)
            prediction_unet = prediction_unet.squeeze().to(bool)
            prediction_unet_pseudo = prediction_unet_pseudo.squeeze().to(bool)

            # evaluate with dice score
            dice_unet = dice(prediction_unet.cpu(), lbl.cpu(), ignore_index=0)
            dice_unet_pseudo = dice(prediction_unet_pseudo.cpu(), lbl.cpu(), ignore_index=0)

            # make step match matplotlib subplot indices
            step = step+1

            # ground truth
            plt.subplot(4, 6, step)
            if step==1:
                plt.title('GROUND TRUTH', fontsize=20, weight='bold', rotation='vertical', x=-0.05, y=0)
            
            plt.text(0.5,1.1,f'{organ.upper()}', weight='bold', fontsize=20, ha='center', va='top',transform=plt.gca().transAxes)
            plt.imshow(color_img, cmap="gray")
            plt.imshow(ground_truth_mask.cpu().numpy(), alpha=0.6, cmap="copper")
            plt.axis("off")

            # SAM prediction
            plt.subplot(4, 6, step+6)
            if step==1:
                plt.title(f'SAM', fontsize=20, weight='bold', rotation='vertical', x=-0.05, y=0)
            plt.imshow(image_orig, cmap="gray")
            show_mask(prediction_sam, plt.gca())

            if prompt in ['point', 'naive_point', 'furthest_from_edges_point', 'one_box_with_points', 'box_and_point', 'box_and_background_point']:
                for input_point, input_label in zip(input_points, input_labels):
                    show_points(input_point, input_label, plt.gca())
            if prompt in ['box', 'one_box_with_points', 'box_and_point', 'box_and_background_point']:
                for input_box in input_boxes:
                    show_box(input_box, plt.gca())
            plt.text(
                0.95,
                0.95,
                f'DSC={dice_sam:.3f}',
                bbox={'facecolor': 'lightblue', 'pad': 10},
                fontsize=16,
                ha='right',
                va='top',
                transform=plt.gca().transAxes
            )
            plt.axis("off")

            # UNET prediction
            plt.subplot(4, 6, step+12)
            if step==1:
                plt.title('UNET', fontsize=20, weight='bold', rotation='vertical', x=-0.05, y=0)
            plt.imshow(image_orig, cmap="gray")
            plt.imshow(prediction_unet.cpu().numpy(), alpha=0.6, cmap="copper")
            plt.text(
                0.95,
                0.95,
                f'DSC={dice_unet:.3f}',
                bbox={'facecolor': 'lightblue', 'pad': 10},
                fontsize=16,
                ha='right',
                va='top',
                transform=plt.gca().transAxes
            )
            plt.axis("off")

            # UNET PSEUDO LABELS prediction
            plt.subplot(4, 6, step+18)
            if step==1:
                plt.title('UNET PSEUDO LABELS', weight='bold', fontsize=20, rotation='vertical', x=-0.05, y=0)
            plt.imshow(image_orig, cmap="gray")
            plt.imshow(prediction_unet_pseudo.cpu().numpy(), alpha=0.6, cmap="copper")
            plt.text(
                0.95,
                0.95,
                f'DSC={dice_unet_pseudo:.3f}',
                bbox={'facecolor': 'lightblue', 'pad': 10},
                fontsize=16,
                ha='right',
                va='top',
                transform=plt.gca().transAxes
            )
            plt.axis("off")
            
            plt.tight_layout()
    
    plt.savefig(f'{output_base_path}example_segmentations_{prompt}.pdf')
    plt.close()




def main():
    device = 'cpu'
    prompt= 'box'

    #NOTE:one image for each organ -- must be  in the same order as config.ORGAN_LIST and of length 6
    image_list = ['liver_76_112.nii.gz', 'lung_031_261.nii.gz', 'pancreas_089_49.nii.gz', 'hepaticvessel_131_32.nii.gz', 'spleen_33_61.nii.gz', 'colon_008_101.nii.gz']
    if len(image_list) != 6:
         ValueError('Wrong length for variable image_list')
    loader = get_loader(image_list)

	# get pretrained SAM model
    model = sam_model_registry['vit_h'](checkpoint=config.SAM_CHECKPOINT_PATH)
    model.to(device)
    predictor = SamPredictor(model)
     
    unet_list = []
    unet_pseudo_labels_list = []
    for organ in config.ORGAN_LIST:
        unet = UNet2D().to(device)
        checkpoint_name = f'unet_{organ.lower()}_all_2D.pth'
        checkpoint = torch.load(os.path.join(config.SAVED_MODEL_PATH, checkpoint_name))
        unet.load_state_dict(checkpoint['model_state_dict'])
        unet_list.append(unet)

        unet_pseudo_labels = UNet2D().to(device)
        checkpoint_name = f'unet_{organ.lower()}_{prompt}_pseudolabels_2D.pth'

        # tasks with two fireground labels in ground truth
        if organ in ['Task03_Liver','Task07_Pancreas', 'Task08_HepaticVessel']:
            checkpoint = torch.load(os.path.join('output/unet/pretrained/', checkpoint_name))
        else:
            checkpoint = torch.load(os.path.join('output/unet/pretrained_old/', checkpoint_name))
        
        unet_pseudo_labels.load_state_dict(checkpoint['model_state_dict'])
        unet_pseudo_labels_list.append(unet_pseudo_labels)

    plot_images(prompt, loader, predictor, unet_list, unet_pseudo_labels_list)


if __name__ == '__main__':
    main()