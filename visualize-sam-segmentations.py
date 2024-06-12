import os
from pathlib import Path
import glob
import random
import torch
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

split = 'test'

output_base_path = config.SAM_OUTPUT_PATH + 'prompt_example_plots/'
good_cases_path = output_base_path + 'best_examples/'
bad_cases_path = output_base_path + 'worst_examples/'
Path(output_base_path).mkdir(parents=True, exist_ok=True)
Path(good_cases_path).mkdir(parents=True, exist_ok=True)
Path(bad_cases_path).mkdir(parents=True, exist_ok=True)


def get_loaders(type):
	loaders = []
	for prompt in config.SAM_PROMPTS_LIST:
		# just to now get box only
		# if prompt!='box':
		# 	continue
		if prompt=='one_box_with_points':	# exclude
			continue
		data_dicts = get_top_images_dict(prompt=prompt, type=type)
		dataset = Dataset(data_dicts, transforms_sam)
		loader = DataLoader(dataset, batch_size=1, num_workers=config.NUM_WORKERS, shuffle=False)
		loaders.append({'prompt': prompt, 'loader':loader})

	return loaders


def get_top_images_dict(prompt, type='top_best'):
	'''
	prompt: one of config.SAM_PROMPT
	type: either top_best or top_worst
	Get one example of a top performing image per task
	'''
	img_dir = os.path.join(config.DATASET_PATH_2D, f"{split}_2d_images")
	lbl_dir = os.path.join(config.DATASET_PATH_2D, f"{split}_2d_masks")
	top_best_dir = config.SAM_OUTPUT_PATH + prompt + '_prompt/' + split + '_images/' + type + '/'
	
	img_fnames = []
	lbl_fnames = []
	names = []

	for organ in config.ORGAN_LIST:
		organ = organ.split('_')[1].lower()

		# choose a random top best image from top_best_dir
		png_names = glob.glob(os.path.join(top_best_dir, f'sam_{organ}*.png'))
		random_png = random.choice(png_names)
		fname = random_png.split('/')[-1].split('.')[0].split('_')
		fname = f'{fname[1]}_{fname[2]}_{fname[3]}.nii.gz'
		img_fname = os.path.join(img_dir, fname)
		img_fnames.append(img_fname)
		lbl_fname = os.path.join(lbl_dir, fname)
		lbl_fnames.append(lbl_fname)
		name = fname.split('.')[0]
		names.append(name)

	data = [{"image": img_fname, "label": lbl_fname, "name": name} for img_fname, lbl_fname, name in zip(img_fnames, lbl_fnames, names)]
		
	print(f'{split} data dict of {format(len(data))} {type} images created for {prompt} prompt')

	return data




def plot_images(loaders, predictor, type):

	# loop through all prompt types & their loaders
	for loader_item in loaders:
		loader = loader_item['loader']
		prompt = loader_item['prompt']

		if prompt not in ['furthest_from_edges_point', 'box_and_background_point']:
			continue
		# if prompt == 'one_box_with_points':
		# 	continue
		print(f'using prompt:{prompt}')

		plt.figure(figsize=(24, 8))
		#plt.suptitle(f"SAM with {prompt}", fontsize=20, weight='bold')

		with torch.no_grad():
			for step, item in enumerate(loader):
				image_orig = item['image'].squeeze()
				ground_truth_mask = item['label'].squeeze().to(bool)
				name = item['name']
				organ = name[0].split('_')[0]

				print(f'plotting for image {name}')

				# convert to rbg uint8 image
				color_img = cv2.cvtColor(image_orig.numpy(), cv2.COLOR_GRAY2RGB)
				color_img = normalize8(color_img)

				# process image to get image embedding
				predictor.set_image(color_img)

				# Predict using point prompts
				if prompt == 'point' or prompt == 'naive_point' or prompt == 'furthest_from_edges_point':
					mask, input_points, input_labels = get_point_prompt_prediction(
						ground_truth_mask=ground_truth_mask,
						predictor=predictor,
						point_type=prompt
					)

				# Predict using bounding box
				if prompt == 'box':
					mask, input_boxes = get_box_prompt_prediction(
						ground_truth_mask=ground_truth_mask,
						predictor=predictor
					)
				
				# Predict using one bounding box and cluster points
				if prompt == 'one_box_with_points':
					mask, input_boxes, input_points, input_labels = get_box_with_points_prediction(
						ground_truth_mask=ground_truth_mask,
						predictor=predictor
					)
				
				# Predict using a box and a point per cluster
				if prompt == 'box_and_point':
					mask, input_boxes, input_points, input_labels = get_box_and_point_prompt_prediction(
						ground_truth_mask=ground_truth_mask,
						predictor=predictor,
						background_point=False
					)
				
				#Predict using a box and a background point per cluster
				if prompt == 'box_and_background_point':
					mask, input_boxes, input_points, input_labels = get_box_and_point_prompt_prediction(
						ground_truth_mask=ground_truth_mask,
						predictor=predictor,
						background_point=True
					)

				dice_pytorch = dice(torch.Tensor(mask).cpu(), ground_truth_mask.cpu(), ignore_index=0)

				# make step match matplotlib subplot indices
				step = step+1

				# ground truth
				plt.subplot(2, 6, step)
				if step==1:
					plt.title('Ground truth', fontsize=16, rotation='vertical', x=-0.05, y=0)
				
				plt.text(0.5,1.1,f'{organ.upper()}', weight='bold', fontsize=16, ha='center', va='top',transform=plt.gca().transAxes)
				plt.imshow(color_img, cmap="gray")
				plt.imshow(ground_truth_mask.cpu().numpy(), alpha=0.6, cmap="copper")
				plt.axis("off")

				# prediction
				plt.subplot(2, 6, step+6)
				if step==1:
					plt.title(f'SAM', fontsize=16, rotation='vertical', x=-0.05, y=0)
				plt.imshow(image_orig, cmap="gray")
				show_mask(mask, plt.gca())

				if prompt in ['point', 'naive_point', 'furthest_from_edges_point', 'one_box_with_points', 'box_and_point', 'box_and_background_point']:
					for input_point, input_label in zip(input_points, input_labels):
						show_points(input_point, input_label, plt.gca())
				if prompt in ['box', 'one_box_with_points', 'box_and_point', 'box_and_background_point']:
					for input_box in input_boxes:
						show_box(input_box, plt.gca())
				plt.text(
					0.95,
					0.95,
					f'DSC={dice_pytorch:.3f}',
					bbox={'facecolor': 'lightblue', 'pad': 10},
					fontsize=16,
					ha='right',
					va='top',
					transform=plt.gca().transAxes
				)

				plt.axis("off")
				
				plt.tight_layout()
		
		if type=='top_best':
			plt.savefig(f'{good_cases_path}{prompt}_{type}.pdf')
		else:
			plt.savefig(f'{bad_cases_path}{prompt}_{type}.pdf')
		plt.close()




def main():
	print(f'Visualizing SAM ')

	# get pretrained SAM model
	model = sam_model_registry['vit_h'](checkpoint=config.SAM_CHECKPOINT_PATH)
	model.to('cpu')
	predictor = SamPredictor(model)

	best_loaders = get_loaders(type='top_best')
	worst_loaders = get_loaders(type='top_worst')

	plot_images(best_loaders, predictor, type='top_best')
	plot_images(worst_loaders, predictor, type='top_worst')



if __name__ == '__main__':
    main()