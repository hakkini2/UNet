import subprocess
from pathlib import Path

import cv2
import fire
import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf
from segment_anything import SamPredictor, sam_model_registry
from torchmetrics.functional.classification import dice

from samDataset import get_loader
from utils.prompts import get_prediction_by_prompt
from utils.utils import normalize8


def main(**kwargs):
    torch.multiprocessing.set_sharing_strategy("file_system")
    from_cli = OmegaConf.create(kwargs)
    base_conf = OmegaConf.load("./configs/_base_config.yaml")
    model_conf = OmegaConf.load("./configs/fm_inference/sam.yaml")
    cfg = OmegaConf.merge(base_conf, model_conf, from_cli)
    assert cfg.prompt in list(cfg.prompts)
    assert cfg.organ in list(cfg.organs)

    # get pretrained SAM model - directly from Meta
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry["vit_h"](checkpoint=cfg.checkpoint)
    model.to(device)
    predictor = SamPredictor(model)

    # get dataloader
    loader = get_loader(cfg)

    # do mask prediction and collect the dice scores
    _ = predict_masks(loader, predictor, cfg=cfg)


def predict_masks(loader, predictor, cfg):
    prompt = cfg.prompt
    organ = cfg.organ

    print(f"Predictions using {prompt} prompt")

    dices = []
    with torch.no_grad():
        for step, item in enumerate(loader):
            print(f"Step {step+1}/{len(loader)}")
            image_orig = item["image"].squeeze()
            ground_truth_mask = item["label"].squeeze().to(bool)
            name = item["name"]

            # convert to rbg uint8 image
            color_img = cv2.cvtColor(image_orig.numpy(), cv2.COLOR_GRAY2RGB)
            color_img = normalize8(color_img)

            # process image to get image embedding
            predictor.set_image(color_img)
            # Predict using input prompt
            pred = get_prediction_by_prompt(ground_truth_mask, predictor, prompt)

            dice_pytorch = dice(torch.Tensor(pred).cpu(), ground_truth_mask.cpu(), ignore_index=0)
            dices.append((name[0], dice_pytorch))

            if cfg.save_pseudo_labels:
                pseudo_masks_path = Path(cfg.data_path) / f"{cfg.split}_2d_{cfg.prompt.replace('/', '_')}_pseudomasks"
                pseudo_masks_path.mkdir(parents=True, exist_ok=True)
                # -- Save pseudomask --
                # modify mask datatype
                mask = np.squeeze(pred).astype(np.float32)
                # save
                mask_img = nib.Nifti1Image(mask, affine=np.eye(4))
                nib.save(mask_img, f"{pseudo_masks_path}/{name[0]}.nii")
                subprocess.run(["gzip", f"{pseudo_masks_path}/{name[0]}.nii"])  # compress

        # sort the dices by the dice score (highest first)
        dices.sort(key=lambda x: x[1].item(), reverse=True)

        # get average dice
        dice_values = list(map(lambda dice: dice[1].item(), dices))
        avg = sum(dice_values) / len(dice_values)
        print(f"Average dice for organ {organ}: {avg:.3f}")

        return dices


if __name__ == "__main__":
    fire.Fire(main)
