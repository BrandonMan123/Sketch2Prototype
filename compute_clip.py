import torch
from torch.nn.functional import normalize
import clip
from PIL import Image
import numpy as np
from torchmetrics.multimodal.clip_score import CLIPScore
from sketch2prototype import load_prompt
import os
import pandas as pd
import gc




def get_image(filepath):
    return np.asarray(Image.open(filepath))

def compute_clip(image, text, metric):
    if not text:
        return 0
    image_tensor = torch.from_numpy(image)
    res =  metric(image_tensor, text).detach().item()
    del image_tensor
    return res

def get_features(image, model, preprocess):
    image = preprocess(Image.open(image)).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    image_features = normalize(image_features, p=2.0, dim=-1)
    return image_features.cpu().detach().numpy()

def compute_image_clip(image1, image2, model, preprocess, device):
    image_1_features = get_features(image1, model, preprocess)
    image_2_features = get_features(image2, model, preprocess)

    similarity = image_2_features @ image_1_features.T
    res = similarity.item() * 100
    return res
    



def compute_clip_from_fp(image_fp, text_fp):
    image = get_image(image_fp)
    image_name = image_fp.split("/")[1]
    text = load_prompt(image_name, text_fp)
    if type(text) != str:
        return 0
    return compute_clip(image, text, metric)

def compute_clip_from_sample(sample_fp, text_fp, model=None, preprocess=None):
    original_clip = compute_clip_from_fp(f"{sample_fp}/original.png", text_fp)
    synthetic_clip = 0
    for i in range(4):
        synthetic_clip += compute_clip_from_fp(f"{sample_fp}/images/image_{i}.png", text_fp)
        # synthetic_clip += compute_image_clip(
        #     f"{sample_fp}/images/image_{i}.png", f"{sample_fp}/original.png")
    synthetic_clip /= 4
    return [original_clip, synthetic_clip]

def pairwise_clip(sample_fp, text_fp, model, preprocess):

    clip_vals = 0
    clip_samples = 0
    for i in range(4):
        for j in range(i+1, 4):
            clip_samples += 1
            clip_vals += compute_image_clip(
                f"{sample_fp}/images/image_{i}.png",
                f"{sample_fp}/images/image_{j}.png",
                model, preprocess, device
            )

    clip_vals /= clip_samples
    return [clip_vals]


def compute_clip_from_dataset(dataset_fp, model, preprocess, ref_file=""):
    original_clips = []
    sample_clips = []
    clip_log = dict()
    if not ref_file:
        ref_df = set()
    else:
        ref_df = set(pd.read_csv(ref_file)['Unnamed: 0'])

    for fp in os.listdir(dataset_fp):

        if fp in ref_df:
            continue

        clips = pairwise_clip(f"{dataset_fp}/{fp}", "data/sketch_drawing.csv", model, preprocess)
        # clips = [1]
        # synthetic_clips = compute_clip_from_sample(f"{dataset_fp}/{fp}", "data/sketch_drawing.csv")
        if clips[0]:
            print (f"{fp}: {clips}")
            clip_log[fp] = clips
            

    return clip_log

def check_dirs_have_image(input_dir):
    no_img = False
    for val in os.listdir(input_dir):
        if "dalle_response.json" not in os.listdir(f"{input_dir}/{val}"):
            print (val)
            no_img = True
    if no_img:
        raise Exception("A directory does not have images")



if __name__ == "__main__":
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    input_dir = "dataset_full"
    device = "cuda"
    save_dir = "clip_scores"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    model, preprocess = clip.load("ViT-B/32", device=device)
    check_dirs_have_image(input_dir)
    clip_log = compute_clip_from_dataset(input_dir, model, preprocess)
    df = pd.DataFrame.from_dict(clip_log, orient="index")
    
    df.to_csv(f"{save_dir}/pairwise_clip.csv")
    print ("Done")
