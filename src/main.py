import json
import multiprocessing
import os
import random

import albumentations as A
import evaluate
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation

import wandb
# Local imports
from config import CHECKPOINT_DIR, HF_REPO, get_device
from data_utils import compute_mean_std
from dataset import BRATSDataset, collate_fn, relevant_slices
from train import train_model

# For ID to label mapping
id2label = {1: "brain", 2: "edema"}

if __name__ == "__main__":
    # If using multiprocessing in DataLoader (num_workers>0), sometimes
    # you might need 'spawn' start method (esp. on Windows).
    multiprocessing.set_start_method("spawn", force=True)

    # Initialize Weights & Biases
    wandb.init(project=os.getenv("WANDB_PROJECT"), name=os.getenv("WANDB_NAME"))

    device = get_device()
    data_dir = os.getenv("DATA_DIR")

    with open(os.path.join(data_dir, "dataset.json"), "r") as f:
        dataset = json.load(f)

    all_train_files = [
        {
            "image": os.path.join(data_dir, d["image"]),
            "label": os.path.join(data_dir, d["label"]),
        }
        for d in dataset["training"]
    ]

    test_files = [{"image": os.path.join(data_dir, d)} for d in dataset["test"]]

    # train-val split
    random.shuffle(all_train_files)
    split_ratio = 0.05
    n_files_for_validation = int(len(all_train_files) * split_ratio)
    val_files = all_train_files[-n_files_for_validation:]
    train_files = all_train_files[:-n_files_for_validation]

    # mean and std accross all images/relevant slices
    mean_train, std_train = compute_mean_std(train_files, relevant_slices)

    # transformations
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[mean_train] * 3, std=[std_train] * 3),
            ToTensorV2(),
        ]
    )

    valtest_transform = A.Compose(
        [A.Normalize(mean=[mean_train] * 3, std=[std_train] * 3), ToTensorV2()]
    )

    # Datasets
    train_dataset = BRATSDataset(train_files, transforms=train_transform)
    val_dataset = BRATSDataset(val_files, transforms=valtest_transform)
    test_dataset = BRATSDataset(test_files, transforms=valtest_transform)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # initialize processor and model
    processor = AutoImageProcessor.from_pretrained(
        "facebook/maskformer-swin-base-coco", use_fast=False, ignore_index=0
    )
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        "facebook/maskformer-swin-base-ade",
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=True,
        num_labels=2,
    )

    # Freeze all parameters in pixel_level_module
    for param in model.model.pixel_level_module.parameters():
        param.requires_grad = False

    # Unfreeze only the decoder and the last stage of the encoder
    for param in model.model.pixel_level_module.decoder.parameters():
        param.requires_grad = True

    for name, param in model.model.pixel_level_module.encoder.named_parameters():
        if "layers.3" in name:
            param.requires_grad = True

    # Metric
    metric = evaluate.load("mean_iou")

    # Train
    train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        preprocessor=processor,
        metric=metric,
        id2label=id2label,
        device=device,
        num_epochs=50,
        learning_rate=5e-5,
        log_interval=10,
        patience=6,
        checkpoint_dir=CHECKPOINT_DIR,
        hf_repo=HF_REPO,
        wandb=wandb,
    )
