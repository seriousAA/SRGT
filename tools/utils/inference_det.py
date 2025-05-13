import torch
from PIL import Image
import os
from pathlib import Path
import random
import numpy as np
from ssod.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.datasets.pipelines.obb.misc import visualize_with_obboxes, vis_args
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import copy
import mmcv

output_path = "/home/****/RS-PCT/data/DOTA/show_scaling"
img_path = '/home/****/RS-PCT/data/DOTA/gsd_dataset/train/P0002_6.png'  # or img = mmcv.imread(img), which will only load it once
output_folder = os.path.join(output_path, os.path.splitext(os.path.basename(img_path))[0])
os.makedirs(output_folder, exist_ok=True)

def process_image(image_path, scale):
    """
    Downsample the image, tile it on a 1024x1024 canvas with random flipping, and save the result.
    """
    try:
        # Load the image
        patch = Image.open(image_path)
        # Calculate the new size
        new_size = int(patch.size[0] * scale), int(patch.size[1] * scale)
        # Resize the image
        small_patch = patch.resize(new_size, Image.ANTIALIAS)
        final_size = 1024

        # Create a new blank image for tiling
        new_patch = Image.new('RGB', (final_size, final_size))
        for i in range(0, final_size, max(1, small_patch.size[0])):
            for j in range(0, final_size, max(1, small_patch.size[1])):
                flip_lr = random.choice([True, False])
                flip_tb = random.choice([True, False])
                tile = small_patch.transpose(Image.FLIP_LEFT_RIGHT) if flip_lr else small_patch
                tile = tile.transpose(Image.FLIP_TOP_BOTTOM) if flip_tb else tile
                new_patch.paste(tile, (i, j))

        # Construct the new file name
        base_name = os.path.basename(image_path)
        name_part, ext_part = os.path.splitext(base_name)
        new_file_name = f"{name_part}_r{scale}{ext_part}"
        new_image_path = os.path.join(output_folder, new_file_name)

        # Save the downsampled and tiled image
        new_patch.save(new_image_path)
        print(f"Saved {new_image_path}")
        return new_image_path

    except Exception as e:
        print(f"Error processing image {image_path} with scale {scale}: {e}")

scales = [0.8, 0.6, 0.4, 0.2]

with ThreadPoolExecutor(max_workers=4) as executor:
    # executor.map returns an iterator that yields results of the function calls
    results = executor.map(partial(process_image, img_path), scales)

# Collect results
processed_image_paths = [img_path]
processed_image_paths.extend(list(results))


config_file = '/home/****/RS-PCT/configs/PseCo/90k_dota1.5_trainval_2.0_trainval_ms.py'
checkpoint_file = '/home/****/RS-PCT/output/1105_sup_trainval_ms/iter_180000.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
results = [inference_detector(model, img)[0] for img in processed_image_paths]

def visualization_func(vis_args, img, result):
    num_classes = len(result)

    # Initialize empty lists to hold the bounding boxes and labels
    all_bboxes = []
    all_labels = []

    # Iterate over each category's results
    for i in range(num_classes):
        # Check if there are detections for the current category
        if result[i].size > 0:
            # Append the bounding boxes to the all_bboxes list
            all_bboxes.append(result[i])
            # Create an array of labels (i) with the same length as the current category's detections
            # and append it to the all_labels list
            all_labels.append(np.full((result[i].shape[0],), i, dtype=np.int64))

    # Concatenate all bounding boxes and labels into two separate arrays
    bboxes = np.concatenate(all_bboxes, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    # Construct the new file name
    base_name = os.path.basename(img)
    name_part, ext_part = os.path.splitext(base_name)
    new_file_name = f"{name_part}_det{ext_part}"
    new_img_path = os.path.join(output_folder, new_file_name)
    vis_args = copy.deepcopy(vis_args)
    vis_args['save_path'] = new_img_path

    visualize_with_obboxes(mmcv.imread(img), bboxes, labels, vis_args)


for img, result in zip(processed_image_paths, results):
    visualization_func(vis_args, img, result)
    print(f"Visualized {img}")