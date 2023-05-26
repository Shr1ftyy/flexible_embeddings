# THIS IS PRETTY SCUFFED LOL
# I'M RUNNING THE SCRIPTS AS SUBPROCESSES xD
# WHY FORK AND MODIFY EXISTING CODE IT'S EASIER TO WORK 
# AROUND IT LIKE THIS?

# import subprocess
import os
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import PurePath
from prompt_script import gen_prompts


parser = argparse.ArgumentParser(description='Run command with arguments.')
parser.add_argument('--config', type=str, required=True, help='Path to base config file')
args = parser.parse_args()

def run_inv_main(base_config, actual_resume, run_name, gpus, data_root, init_word, **kwargs):
    cmd = [
        'python', 'main.py', 
        '--base', base_config, 
        '-t',
        '--actual_resume', actual_resume,
        '-n', run_name,
        '--gpus', gpus, # TODO: GPUs
        '--data_root', data_root,
        '--init_word', init_word
    ]

    print("TRAINING PARAMS:")
    print(" ".join(cmd))

    for key, value in kwargs.items():
        cmd.append(key)
        cmd.append(value)

    os.system(" ".join(cmd))

def run_txt2img(ddim_eta, n_samples, n_iter, scale, ddim_steps, embedding_path, ckpt_path, prompt, outdir, **kwargs):
    cmd = [
        'python', 'scripts/txt2img.py',
        '--ddim_eta', str(ddim_eta),
        '--n_samples', str(n_samples),
        '--n_iter', str(n_iter),
        '--scale', str(scale),
        '--ddim_steps', str(ddim_steps),
        '--embedding_path', embedding_path,
        '--ckpt_path', ckpt_path,
        '--prompt', prompt,
        '--outdir', outdir
    ]

    print("PARAMS:")
    print(" ".join(cmd))

    for key, value in kwargs.items():
        cmd.append(key)
        cmd.append(value)

    os.system(" ".join(cmd))

    

if __name__ == "__main__":

    conf = OmegaConf.load(args.config)

    embedding_names = conf.embedding_names
    base_config = conf.base_config
    model_ckpt_path = conf.model_ckpt_path
    init_img_path = conf.init_img_path
    iterations = conf.iterations
    outputs_path = conf.outputs_path
    init_word = conf.init_word

    for name in embedding_names:
        for iteration in tqdm(range(iterations)):
            data_root = ''

            # set data root prefix (for images) to outputs folder if iteration
            # is more than 0, else set it to the init_images folder :D

            if iteration > 0:
                # TODO: change output_paths to filtered_paths, and iteration when adding image filtering
                # data_root += str(PurePath(f"{filtered_paths}/{name}-{iteration}/samples/"))
                data_root += str(PurePath(f"{outputs_path}/{name}-{iteration-1 if iteration > 0 else ''}/samples/"))
            else:
                data_root += str(PurePath(f"{init_img_path}/{name}/"))
            

            # train embedding
            run_inv_main(
                base_config=base_config,
                actual_resume=model_ckpt_path, 
                run_name=f"{name}-{iteration}", 
                gpus=conf.gpus,
                data_root=data_root, 
                init_word=init_word # TODO: This is constant, change later
            )

            # generate prompts and place output images into outputs_path
            # TODO: name = "_" + name , have to have underscore before name bcos of logging stuff in main.py 
            # TODO: change output image params
            prompts = gen_prompts(
                num_prompts=conf.num_prompts, 
                num_attire=conf.num_attire, 
                subject="*", # main.py script automatically replaces this with embedding
                simple=conf.simple_attire
            )

            for prompt in prompts:
                run_txt2img(
                    ddim_eta=conf.ddim_eta, 
                    n_samples=conf.n_samples, 
                    n_iter=conf.n_iter, 
                    scale=conf.scale, 
                    ddim_steps=conf.ddim_steps, 
                    embedding_path=f"logs/_{name}-{iteration}/checkpoints/embeddings.pt",
                    ckpt_path=model_ckpt_path, 
                    # prompt='\"a photo of *\"', # TODO: a prompt here from prompt generator
                    prompt=f'\"{prompt}\"',
                    outdir=f'{outputs_path}/{name}-{iteration}/'
                )   

        # filter images, and output them to filtered_paths

        # repeat 

    args = parser.parse_args()

'''
python main.py --base configs/test.yaml
               -t 
               --actual_resume /content/drive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/model.ckpt
               -n TST_NAME
               --gpus 0, 
               --data_root init_images/example
               --init_word "*"
'''

'''
python scripts/txt2img.py --ddim_eta 0.0 
                          --n_samples 8 
                          --n_iter 1
                          --scale 7.0
                          --ddim_steps 20 
                          --embedding_path /content/drive/MyDrive/TopicsInCS/textual_inversion/logs/example2023-05-26T11-31-52_TST_NAME/checkpoints/embeddings_gs-9.pt
                          --ckpt_path ~/model.ckpt 
                          --prompt "a photo of *"
'''