from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset
from transformers import MaskFormerImageProcessor

from data_utils import load_flair, load_mask_brain_edema, normalize_img

# Slices (by index) to extract from each volume
relevant_slices = range(52, 105)


@dataclass
class BRATDataInput:
    original_image: np.ndarray
    transformed_image: np.ndarray
    original_segmentation_map: np.ndarray
    transformed_segmentation_map: np.ndarray


class BRATSDataset(Dataset):
    """
    PyTorch dataset for the Brain Tumor segmentation task
    (BRATS-like structure).
    """

    def __init__(self, data_list, transforms=None, slice_axis=2):
        """
        Args:
            data_list: List[Dict]: each with {'image': path, 'label': path}.
            transforms: Albumentations transforms.
            slice_axis: axis to perform the slicing on (default: 2)
        """
        self.data_list = data_list
        self.transforms = transforms
        self.slice_axis = slice_axis
        self.slices_to_retrieve = relevant_slices

    def __len__(self):
        return len(self.data_list) * len(self.slices_to_retrieve)

    def __getitem__(self, idx):
        scan_idx = idx // len(self.slices_to_retrieve)
        slice_idx = self.slices_to_retrieve[idx % len(self.slices_to_retrieve)]

        sample = self.data_list[scan_idx]

        # load flair slice and normalize
        image_data = load_flair(sample, slice_idx)  # shape: (240, 240)
        image_data = normalize_img(image_data)  # shape: (240, 240)

        # convert to 3 channels
        image_data = np.stack([image_data] * 3, axis=-1).astype(np.float32)

        # load segmentation mask
        label_data = load_mask_brain_edema(sample, slice_idx).astype(np.uint8)

        # transformations
        if self.transforms:
            augmented = self.transforms(image=image_data, mask=label_data)
            transformed_image = augmented["image"]
            transformed_segmentation_map = augmented["mask"]
        else:
            transformed_image, transformed_segmentation_map = image_data, label_data

        return BRATDataInput(
            original_image=image_data,
            transformed_image=transformed_image,
            original_segmentation_map=label_data,
            transformed_segmentation_map=transformed_segmentation_map,
        )


def collate_fn(batch):
    """
    Collate function to prepare a batch for the MaskFormerImageProcessor.
    """

    preprocessor = MaskFormerImageProcessor(
        ignore_index=-1,
        do_reduce_labels=False,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )

    original_images = [sample.original_image for sample in batch]
    transformed_images = [sample.transformed_image for sample in batch]
    original_segmentation_maps = [sample.original_segmentation_map for sample in batch]
    transformed_segmentation_maps = [
        sample.transformed_segmentation_map for sample in batch
    ]

    preprocessed_batch = preprocessor(
        transformed_images,
        segmentation_maps=transformed_segmentation_maps,
        return_tensors="pt",
    )

    preprocessed_batch["original_images"] = original_images
    preprocessed_batch["original_segmentation_maps"] = original_segmentation_maps

    return preprocessed_batch
