import os
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import PurePath
from prompt_script import gen_prompts
import cv2
import clip
import torchvision.transforms.functional as TF
import torch
from torchvision import transforms
import time
import logging

parser = argparse.ArgumentParser(description='Run command with arguments.')
parser.add_argument('--config', type=str, required=True, help='Path to base config file')
parser.add_argument('--debug', type=str, required=True, help='Path to base config file')

args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)

device = "cuda" if torch.cuda.is_available() else "cpu"

def transform_image(image):
    # Resize the image
    image = TF.resize(image, (224, 224))
    # Normalize the image
    image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

def filter_images(model, init_img, to_filter, filter_ratio):
    # Prepare the init_img
    init_img = transform_image(init_img)

    # init_img = normalize(init_img, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    init_img = init_img.unsqueeze(0).to(device)
    init_img_features = model.encode_image(init_img)

    # Calculate CLIP score for each image in to_filter and store them in scores
    scores = []
    for img in to_filter:
        # img = normalize(img, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        img = transform_image(img)

        img = img.unsqueeze(0).to(device)
        img_features = model.encode_image(img)

        # Calculate similarity score using cosine similarity
        score = torch.nn.functional.cosine_similarity(init_img_features, img_features, dim=-1)
        scores.append(score.item())

    # Get the top filter_ratio percentage of images
    top_k = int(len(to_filter) * filter_ratio)
    top_k_indices = torch.topk(torch.Tensor(scores), k=top_k).indices

    return [to_filter[i] for i in top_k_indices]

def load_image(img_path):
    # OpenCV loads an image as BGR format. So, we need to convert it to RGB
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    # Convert the image from a NumPy array to a PyTorch tensor and normalize pixel values
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Converts a NumPy array to a PIL Image
        transforms.ToTensor(),  # Converts a PIL Image to a PyTorch tensor
    ])

    img_t = transform(img)
    return img_t


def save_images(image_tensors, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Transformation to convert tensor to PIL Image
    to_pil = transforms.ToPILImage()

    for i, img_tensor in enumerate(image_tensors):
        # Convert the tensor to a PIL Image
        img = to_pil(img_tensor)

        # Define the output path
        out_path = os.path.join(output_dir, f'image_{i}.jpg')

        # Save the image
        img.save(out_path)

def get_all_file_paths(directory):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # Combine the directory path and file name
            file_path = os.path.join(dirpath, filename)
            file_paths.append(file_path)
    return file_paths


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

    logging.debug("TRAINING PARAMS:")
    logging.debug(" ".join(cmd))

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

    logging.debug("GEN PARAMS:")
    logging.debug(" ".join(cmd))

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
    filtered_path = conf.filtered_path
    clip_encoder, transform = clip.load("ViT-L/14", device=device) # TODO: make this user-defined in the future?

    for name in embedding_names:
        for iteration in tqdm(range(iterations)):
            data_root = ''

            # set data root prefix (for images) to outputs folder if iteration
            # is more than 0, else set it to the init_images folder :D

            if iteration > 0:
                # TODO: change output_paths to filtered_paths, and iteration when adding image filtering
                data_root += str(PurePath(f"{filtered_path}/{name}-{iteration-1 if iteration > 0 else ''}/samples/"))
                # data_root += str(PurePath(f"{outputs_path}/{name}-{iteration-1 if iteration > 0 else ''}/samples/"))
            else:
                data_root += str(PurePath(f"{init_img_path}/{name}/"))
            
s
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

            outdir = str(PurePath(f'{outputs_path}/{name}-{iteration}/'))

            for prompt in prompts:
                run_txt2img(
                    ddim_eta=conf.ddim_eta, 
                    n_samples=conf.n_samples, 
                    n_iter=conf.n_iter, 
                    scale=conf.scale, 
                    ddim_steps=conf.ddim_steps, 
                    embedding_path=f"logs/_{name}-{iteration}/checkpoints/embeddings.pt",
                    ckpt_path=model_ckpt_path, 
                    # prompt='\"a photo of *\"',
                    # TODO: look into better prompt generators
                    prompt=f'\"{prompt}\"',
                    outdir=outdir
                )   

            output_samples_path = str(PurePath(outdir + '/samples'))
            # set up next training image batch
            # get initial image
            init_img_file_path = get_all_file_paths(f'{init_img_path}/{name}')[0]
            init_img = load_image(init_img_file_path)
            logging.debug("INIT IMG:")
            logging.debug(init_img)

            # load images to filter into mem
            images_to_filter = []

            output_imgs = get_all_file_paths(output_samples_path)
            for img_path in output_imgs:
                out_img = load_image(img_path)
                images_to_filter.append(out_img)
            logging.debug("IMAGES TO FILTER:")
            logging.debug(images_to_filter)

            # filter images, and output them to filtered_paths
            filtered_images = filter_images(
                model=clip_encoder,
                init_img=init_img,
                to_filter=images_to_filter,
                filter_ratio=conf.filter_ratio,
            )
            logging.debug("FILTERED IMAGES:")
            logging.debug(filtered_images)
            time.sleep(2)

            save_images(filtered_images, str(PurePath(f"{filtered_path}/{name}-{iteration}/samples")))

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
