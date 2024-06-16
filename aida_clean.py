import os
import json
import random

import shutil

from tqdm import tqdm

import numpy as np

import ast
from PIL import Image
from io import BytesIO


INPUT_DIR = "data/Aida"
OUTPUT_DIR = "data/cleaned_aida_b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 101
random.seed(SEED)
np.random.seed(SEED)


# pre-sample the image indices for each batch
N_IMG_PER_BATCH = 10_000
N_SAMPLES = 1_000

f_batch_sampled_indices = f"{OUTPUT_DIR}/batch_sampled_indices.json"
if os.path.exists(f_batch_sampled_indices):
    with open(f_batch_sampled_indices) as f:
        batch_sampled_indices = json.load(f)
else:
    batch_sampled_indices = {
        f"batch_{i}": np.sort(np.random.choice(N_IMG_PER_BATCH, size=N_SAMPLES, replace=False)).tolist()
        for i in range(1, 11)
    }
    with open(f_batch_sampled_indices, mode='w') as f:
        json.dump(batch_sampled_indices, f, indent=4)


def string_to_bytes(string):
    return ast.literal_eval(string)

def parse_masks(png_mask_strs):
    masks = []
    for mask_str in png_mask_strs:
        mask_bytes = string_to_bytes(mask_str)
        mask = Image.open(BytesIO(mask_bytes))
        mask = mask.convert("L")
        masks.append(mask.getdata())
    
    return masks

def construct_masked_image(image_data):
    w, h = image_data["width"], image_data["height"]
    masks = parse_masks(image_data["png_masks"])

    masked_image = Image.new(mode="L", size=(w, h))
    base_mask = [0 for _ in range(w * h)]
    for mask in masks:
        for idx, m in enumerate(mask):
            if m != 0:
                base_mask[idx] = m
    masked_image.putdata(base_mask)
    
    return masked_image

def check_completeness(image_id: str):
    img_output_dir = f"{OUTPUT_DIR}/{image_id}"
    raw_img_exist = os.path.exists(f"{img_output_dir}/raw_image.jpg")
    msk_img_exist = os.path.exists(f"{img_output_dir}/masked_image.png")
    metadata_exist = os.path.exists(f"{img_output_dir}/metadata.json")
    
    return (raw_img_exist and msk_img_exist and metadata_exist)

def process_data_batch(batch_id):
    subdir = f"batch_{batch_id}"
    print(f"Processing {subdir} ...")

    with open(f"{INPUT_DIR}/{subdir}/JSON/kaggle_data_{batch_id}.json") as f:
        data = json.load(f)

    sampled_indices = batch_sampled_indices[subdir]
    pbar = tqdm(sampled_indices)
    for i in pbar:
        image_id = data[i]['uuid']
        
        pbar.set_postfix_str(f"{i}: {image_id}")
        
        img_output_dir = f"{OUTPUT_DIR}/{image_id}"
        if check_completeness(image_id):
            continue
        elif os.path.exists(img_output_dir):
            print(f"Warn: {img_output_dir} exists but not completed.")
            shutil.rmtree(img_output_dir)
            
        os.makedirs(img_output_dir, exist_ok=True)
        
        # save raw image
        shutil.copyfile(f"{INPUT_DIR}/{subdir}/background_images/{image_id}.jpg", f"{img_output_dir}/raw_image.jpg")
        
        # save masked image
        image_mask = construct_masked_image(data[i]['image_data'])
        image_mask.save(f"{img_output_dir}/masked_image.png")
        
        # save metadata
        with open(f"{img_output_dir}/metadata.json", mode='w') as f:
            json.dump(data[i], f, indent=4)


if __name__ == "__main__":
    for i in range(1, 11):
        process_data_batch(i)