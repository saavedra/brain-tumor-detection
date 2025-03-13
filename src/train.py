import json
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def load_checkpoint(
    model, optimizer=None, checkpoint_dir="checkpoints", metric="weighted_iou"
):
    """
    Load the best-performing checkpoint based on a selected metric.

    Args:
        model (torch.nn.Module): Model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer instance to restore state (if needed).
        checkpoint_dir (str): Directory where checkpoints are stored.
        metric (str): The metric used to determine the best model (e.g., "mean_iou", "weighted_iou").

    Returns:
        model (torch.nn.Module): Model with loaded weights.
        start_epoch (int): The epoch to resume training from.
        best_metric_value (float): The best value of the chosen metric.
    """
    metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")

    if not os.path.exists(metadata_path):
        print("No checkpoint metadata found. Starting fresh.")
        return model, 1, 0.0  # Start fresh

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Find the best checkpoint based on the selected metric
    best_checkpoint = None
    best_metric_value = float("-inf")  # Initialize as lowest possible value

    for checkpoint in metadata.get("checkpoints", []):
        if metric in checkpoint["scores"]:  # Ensure metric exists
            metric_value = checkpoint["scores"][metric]
            if metric_value > best_metric_value:  # Find max value
                best_metric_value = metric_value
                best_checkpoint = checkpoint

    if not best_checkpoint:
        print(f"No valid checkpoint found with metric '{metric}'. Starting fresh.")
        return model, 1, 0.0

    best_checkpoint_path = best_checkpoint["checkpoint_path"]
    start_epoch = best_checkpoint["epoch"] + 1  # Resume from the next epoch

    print(
        f"Loading best checkpoint from {best_checkpoint_path} (Epoch {best_checkpoint['epoch']} with {metric} = {best_metric_value:.4f})"
    )

    # Load model state
    model.load_state_dict(torch.load(best_checkpoint_path))

    # Load optimizer state if available
    optimizer_path = best_checkpoint_path.replace(".pth", "_optimizer.pth")
    if optimizer and os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))
        print("Optimizer state loaded.")

    return model, start_epoch, best_metric_value


def save_checkpoint(model, epoch, scores: dict, checkpoint_dir="checkpoints"):
    """
    Save all model checkpoints along with metadata.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")

    # Save model weights
    torch.save(model.state_dict(), checkpoint_path)

    # Convert NumPy arrays to lists
    for key, value in scores.items():
        if isinstance(value, np.ndarray):
            scores[key] = value.tolist()

    checkpoint_metadata = {
        "epoch": epoch,
        "scores": scores,
        "checkpoint_path": checkpoint_path,
    }

    # Load previous metadata if exists
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"checkpoints": []}

    # Append new checkpoint metadata
    metadata["checkpoints"].append(checkpoint_metadata)

    # Save updated metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Checkpoint saved at {checkpoint_path} (Epoch {epoch})")


def evaluate_model(
    model, dataloader, preprocessor, metric, id2label, device, max_batches=None
):
    """
    Evaluate the model on a given dataloader.
    """
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and idx >= max_batches:
                break

            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)

            # Resize back to original shape
            original_images = batch["original_images"]
            target_sizes = [(img.shape[0], img.shape[1]) for img in original_images]

            # Post-process
            predicted_segmentation_maps = (
                preprocessor.post_process_semantic_segmentation(
                    outputs, target_sizes=target_sizes
                )
            )
            ground_truth_segmentation_maps = batch["original_segmentation_maps"]

            metric.add_batch(
                references=ground_truth_segmentation_maps,
                predictions=predicted_segmentation_maps,
            )

    metric_score = metric.compute(num_labels=len(id2label), ignore_index=None)
    return metric_score


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    preprocessor,
    metric,
    id2label,
    device,
    num_epochs=50,
    learning_rate=5e-5,
    weight_decay=5e-5,
    log_interval=100,
    patience=5,
    checkpoint_dir="checkpoints",
    hf_repo=None,
    wandb=None,
):
    """
    Train the MaskFormer model, resuming from the last checkpoint if available.
    """
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Resume from checkpoint if available
    model, start_epoch, best_val_iou = load_checkpoint(model, optimizer, checkpoint_dir)

    epochs_without_improvement = 0

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n===== EPOCH {epoch}/{num_epochs} =====")
        model.train()
        running_loss = 0.0

        for idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            optimizer.zero_grad()

            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[ml.to(device) for ml in batch["mask_labels"]],
                class_labels=[cl.to(device) for cl in batch["class_labels"]],
            )

            loss = outputs.loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (idx % log_interval == 0) and (idx > 0):
                avg_loss = running_loss / (idx + 1)
                print(f"Epoch {epoch}, Iter {idx} - Loss: {avg_loss:.6f}")
                # print(f"Model loss: {ce_loss.item()}, class loss: {class_loss.item()}\n")
                if wandb:
                    wandb.log(
                        {"Training Loss": avg_loss, "Epoch": epoch, "Iteration": idx}
                    )

        # Evaluate on validation subset
        val_scores = evaluate_model(
            model=model,
            dataloader=val_dataloader,
            preprocessor=preprocessor,
            metric=metric,
            id2label=id2label,
            device=device,
            max_batches=6,
        )

        # identifying edema is more important than brain
        val_weighted_iou = (
            val_scores["per_category_iou"][0] * 0.3
            + val_scores["per_category_iou"][1] * 0.7
        )
        val_scores["weighted_iou"] = val_weighted_iou

        print(f"Validation Weighted IoU: {val_weighted_iou}")
        print(f"Validation scores:", val_scores)

        if wandb:
            wandb.log(val_scores)

        # save checkpoint
        save_checkpoint(model, epoch, val_scores, checkpoint_dir)

        # did performance improve?
        if val_weighted_iou > best_val_iou:
            best_val_iou = val_weighted_iou
            epochs_without_improvement = 0

            # Push to HuggingFace Hub
            if hf_repo is not None:
                model.push_to_hub(
                    hf_repo, commit_message=f"Best model update - Epoch {epoch}"
                )
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs of no improvement."
                )
                break

        scheduler.step()

    print("Training complete.")
    if wandb:
        wandb.finish()
