import torch
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

def compute_clip(image, text):
    if not text:
        return 0
    return metric(torch.from_numpy(image), text).detach().item()

def compute_image_clip(image1, image2):
    image_1 = preprocess(Image.open(image1)).unsqueeze(0).to(device)
    image_2 = preprocess(Image.open(image2)).unsqueeze(0).to(device)

    image_1_features = model.encode_image(image_1)
    image_1_features /= image_1_features.norm(dim=-1, keepdim=True)

    image_2_features = model.encode_image(image_2)
    image_2_features /= image_2_features.norm(dim=-1, keepdim=True)

    similarity = image_2_features.cpu().detach().numpy() @ image_1_features.cpu().detach().numpy().T
    res = similarity.item() * 100
    del similarity
    del image_1_features
    del image_2_features
    del image_1
    del image_2
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated())
    return res
    



def compute_clip_from_fp(image_fp, text_fp):
    image = get_image(image_fp)
    image_name = image_fp.split("/")[1]
    text = load_prompt(image_name, text_fp)
    if type(text) != str:
        return 0
    return compute_clip(image, text)

def compute_clip_from_sample(sample_fp, text_fp):
    original_clip = compute_clip_from_fp(f"{sample_fp}/original.png", text_fp)
    synthetic_clip = 0
    for i in range(4):
        synthetic_clip += compute_clip_from_fp(f"{sample_fp}/images/image_{i}.png", text_fp)
        # synthetic_clip += compute_image_clip(
        #     f"{sample_fp}/images/image_{i}.png", f"{sample_fp}/original.png")
    synthetic_clip /= 4
    return [original_clip, synthetic_clip]

def pairwise_clip(sample_fp, text_fp):

    clip_vals = 0
    clip_samples = 0
    for i in range(4):
        for j in range(i+1, 4):
            clip_samples += 1
            clip_vals += compute_image_clip(
                f"{sample_fp}/images/image_{i}.png",
                f"{sample_fp}/images/image_{j}.png"
            )

    clip_vals /= clip_samples
    return [clip_vals]


def compute_clip_from_dataset(dataset_fp, ref_file=""):
    original_clips = []
    sample_clips = []
    clip_log = dict()
    if not ref_file:
        ref_df = set()
    else:
        ref_df = set(pd.read_csv(ref_file)['Unnamed: 0'])

    for fp in os.listdir(dataset_fp):

        if fp in ref_df:
            print ("Hi")
            continue

        clips =\
        pairwise_clip(f"{dataset_fp}/{fp}", "data/sketch_drawing.csv")
        if clips[0]:
            print (f"{fp}: {clips}")
            clip_log[fp] = clips
            torch.cuda.synchronize() 
            

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
    device = "cpu"
    save_dir = "clip_scores/pairwise_clip_tmp.csv"

    model, preprocess = clip.load("ViT-B/32", device=device)
    check_dirs_have_image(input_dir)
    clip_log = compute_clip_from_dataset(input_dir)
    df = pd.DataFrame.from_dict(clip_log, orient="index")
    
    df.to_csv(save_dir)
    print ("Done")