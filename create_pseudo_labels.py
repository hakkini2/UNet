import subprocess

from omegaconf import OmegaConf

if __name__ == "__main__":
    cfg = OmegaConf.load("./configs/_base_config.yaml")

    prompts_list = ["box_and_then_point", "box_and_then_background_point", "box_and_then_fg/bg_point"]
    for organ in list(cfg.organs):
        for prompt in prompts_list:
            assert prompt in list(cfg.prompts)
            print("Generating pseudo labels for ", organ, " with ", prompt)
            subprocess.run(
                [
                    "python",
                    "sam_eval.py",
                    "--organ",
                    organ,
                    "--prompt",
                    prompt,
                    "--split",
                    "train",
                    "--save_pseudo_labels",
                ]
            )
