import config
import subprocess

if __name__ == '__main__':

    prompts_list = ['box_and_then_point', 'box_and_then_background_point', 'box_and_then_fgorbg_point']
    for organ in config.ORGAN_LIST:
        for prompt in prompts_list:
            assert prompt in config.SAM_PROMPTS_LIST
            print("Generating pseudo labels for ", organ, " with ", prompt)
            subprocess.run(["python", "sam-pseudo-labels.py", "--organ", organ, "--prompt", prompt]) 
