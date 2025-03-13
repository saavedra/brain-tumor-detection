import os
import random

import nibabel as nib
import numpy as np


def load_nii(path):
    return nib.load(path).get_fdata()


def normalize_img(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)


def load_flair(sample, slice_idx=None):
    return load_nii(sample["image"])[:, :, slice_idx, 0]


def load_mask_brain_edema(sample, slice_idx):
    mask_flair = load_flair(sample, slice_idx)
    mask_labels = load_nii(sample["label"])[:, :, slice_idx]

    # 0: background, 1: brain, 2: edema
    return np.where(mask_labels >= 1, 2, np.where(mask_flair >= 1, 1, 0))


def compute_mean_std(data_list, slice_indices, use_precomputed = True):
    """
    Compute mean and std for the entire dataset across specified slices.
    """
    if use_precomputed: 
        return 128.21144104003906, 255.01385498046875 # pre-computed values
    sum_pixels = 0
    sum_sq_pixels = 0
    total_pixels = 0

    print(f'Computing mean and std from {len(data_list) * len(slice_indices)} examples')

    for sample in data_list:
        image_data = np.stack(
            [load_flair(sample, idx) for idx in slice_indices], axis=0
        )
        sum_pixels += image_data.sum()
        sum_sq_pixels += (image_data**2).sum()
        total_pixels += image_data.size

    mean = sum_pixels / total_pixels
    std = np.sqrt((sum_sq_pixels / total_pixels) - (mean**2))
    print(f'mean: {mean}, std: {std}')
    return np.float32(mean), np.float32(std)


def denormalize(image_tensor, mean_train, std_train):
    image_np = image_tensor.detach().cpu().numpy()
    image_np = image_np[0]  # (C, H, W) -> (H, W)

    # Reverse normalization
    image_np = (image_np * std_train) + mean_train

    # Rescale to 0-1
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-6)

    return (image_np * 255).astype(np.uint8)  # Convert to uint8
