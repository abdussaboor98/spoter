# SPOTER Repository Guide

This document summarizes the structure of the SPOTER (Sign POse-based TransformER) repository and explains how data flows through the project, how training is orchestrated, and how the core model works. Use it as a quick reference when navigating or extending the codebase.

## High-level Overview

The repository implements SPOTER, a Transformer-based architecture for word-level sign language recognition that operates on sequences of 2D skeletal keypoints extracted from videos. The workflow is:

1. **Data ingestion** via CSV files containing frame-wise body and hand joint coordinates.
2. **Optional augmentation** and **pose normalization** to improve robustness.
3. **Sequence modeling** with a lightweight Transformer encoder-decoder variant that maps pose sequences to class logits.
4. **Training orchestration** with checkpointing, evaluation, and optional plotting utilities.

## Directory Structure

- `augmentations/`: Pose-specific augmentation operators for rotation, shearing, and arm articulation perturbations.【F:augmentations/__init__.py†L1-L205】
- `datasets/`: Dataset wrappers that load CSV pose annotations and apply augmentation/normalization pipelines before handing tensors to the model.【F:datasets/czech_slr_dataset.py†L1-L116】
- `normalization/`: Boháček-style normalization routines for torso and hand joints that rescale poses into canonical bounding boxes per frame.【F:normalization/body_normalization.py†L1-L207】【F:normalization/hand_normalization.py†L1-L125】
- `spoter/`: Core model, noise regularization, and training utilities such as epoch loops and evaluation helpers.【F:spoter/spoter_model.py†L1-L69】【F:spoter/utils.py†L1-L67】【F:spoter/gaussian_noise.py†L1-L16】
- `train.py`: Entry point for training, handling argument parsing, seeding, dataloaders, optimizer/scheduler setup, checkpoint evaluation, and result visualization.【F:train.py†L1-L213】
- `utils.py`: Helper functions for dataset splitting and statistics logging used during experiment configuration.【F:utils.py†L1-L34】

## Data Pipeline

### Source Format

CSV files contain stringified lists of coordinates for each joint and frame. Column names follow `joint_side_coord` (e.g., `leftWrist_X`) conventions that the loader adapts to the internal identifier scheme. Each row represents a signing instance with synchronized pose sequences and a `labels` column holding class IDs.【F:datasets/czech_slr_dataset.py†L1-L49】

### Loading and Tensors

`CzechSLRDataset` (despite the name, it supports any dataset with the same schema) reads CSV rows into NumPy arrays shaped `[frames, joints, 2]`. During `__getitem__`, the sample is converted to a dictionary keyed by joint identifiers so the augmentation and normalization routines can operate on semantically meaningful structures before being converted back into tensors.【F:datasets/czech_slr_dataset.py†L51-L112】

### Augmentations

Augmentations are probabilistic and executed per sample when `augmentations=True`:
- **Global rotation**: Rotates every keypoint around the frame center by a random angle (±13°).【F:augmentations/__init__.py†L61-L110】
- **Shear/perspective**: Applies either horizontal squeezing or a perspective warp to emulate viewpoint changes.【F:augmentations/__init__.py†L112-L181】
- **Arm joint rotation**: Traverses arm joints and randomly rotates downstream joints to mimic natural variations in articulation.【F:augmentations/__init__.py†L183-L231】

### Normalization

Normalization ensures body and hand coordinates occupy consistent canonical boxes even when the original detector misses joints:
- **Body** normalization estimates a torso-centered bounding box per frame using shoulder or neck/nose distances (head metric) and rescales each joint into `[0,1]` coordinates. Missing reference joints fall back to the previous valid frame.【F:normalization/body_normalization.py†L17-L205】
- **Hand** normalization isolates each hand, computes per-frame bounding boxes with adaptive padding, and rescales coordinates, supporting one or two hands depending on availability.【F:normalization/hand_normalization.py†L23-L123】

### Final Tensor Adjustments

After normalization, coordinates are converted back to tensors and shifted by `-0.5` to center the distribution around zero, which benefits Transformer training. Optional Gaussian noise (mean 0, std configurable) can be injected to the tensor for regularization via `spoter.gaussian_noise.GaussianNoise`.【F:datasets/czech_slr_dataset.py†L101-L116】【F:spoter/gaussian_noise.py†L1-L16】

## Model Architecture

`spoter.spoter_model.SPOTER` constructs a standard `nn.Transformer` with 6 encoder and 6 decoder layers, 9 attention heads, and configurable hidden size (default 108). Key architectural tweaks:

- **Positional Encoding**: `self.pos` is a learned embedding added to the flattened input sequence before feeding it to the Transformer encoder.【F:spoter/spoter_model.py†L31-L40】
- **Class Query Token**: `self.class_query` acts as the sole decoder input token that queries the encoded pose sequence, analogous to a CLS token.【F:spoter/spoter_model.py†L31-L40】
- **Decoder Simplification**: `SPOTERTransformerDecoderLayer` removes the standard decoder self-attention, keeping only cross-attention over encoder memory and the feed-forward blocks, reflecting the single-query design.【F:spoter/spoter_model.py†L11-L37】
- **Classification Head**: A linear layer maps the decoder output to `num_classes` logits per sequence.【F:spoter/spoter_model.py†L34-L43】

The forward pass flattens spatial dimensions, injects positional embeddings, runs the Transformer, and produces class logits. The tensor is unsqueezed to maintain batch semantics during evaluation utilities.【F:spoter/spoter_model.py†L41-L48】

## Training Loop

`train.py` orchestrates experiments:

1. **Argument parsing**: `get_default_args()` exposes hyperparameters for datasets, optimization, scheduling, augmentation noise, and visualization with sensible defaults.【F:train.py†L17-L84】
2. **Reproducibility**: Seeds Python, NumPy, and PyTorch RNGs (including CUDA) and enforces deterministic CuDNN behavior.【F:train.py†L87-L104】
3. **Logging**: Writes progress to `<experiment_name>_<split>.log` using Python logging and prints to stdout.【F:train.py†L105-L118】
4. **Model & Optimizer**: Instantiates `SPOTER`, CrossEntropy loss, SGD optimizer, and an optional ReduceLROnPlateau scheduler.【F:train.py†L119-L142】
5. **DataLoaders**: Builds train/val/test loaders with optional stratified splits and augmentation transforms. `__balance_val_split` and `__split_of_train_sequence` in `utils.py` manage stratified sampling and subset shrinking for ablation studies.【F:train.py†L145-L196】【F:utils.py†L1-L24】
6. **Epoch Loop**: Uses `spoter.utils.train_epoch` to run backprop per epoch, collecting loss/accuracy. Validation uses `spoter.utils.evaluate` with model.eval toggling.【F:train.py†L198-L225】【F:spoter/utils.py†L1-L45】
7. **Checkpointing**: Saves best training and validation checkpoints every ten epochs. Later, all checkpoints are reloaded to evaluate on the held-out test set, printing per-checkpoint accuracy and tracking the best result.【F:train.py†L226-L273】
8. **Visualization**: Optionally plots loss/accuracy curves and LR schedules to `out-img/` for experiment logging.【F:train.py†L275-L310】

`train_epoch` and `evaluate` expect dataloaders that yield batches shaped `[1, frames, joints, 2]`. They squeeze singleton batch dimensions, run the model, compute softmax predictions, and tally accuracies. `evaluate_top_k` provides k-top accuracy support when needed.【F:spoter/utils.py†L1-L67】

## Experiment Tips

- **Dataset Preparation**: Ensure CSVs match the expected column schema (body and hand identifiers). Missing joints default to zeros; normalization logic handles fallback but consistent detection quality improves accuracy.
- **Hyperparameters**: The default hidden dimension (108) reflects the WACV 2022 setup. Adjust `--hidden_dim`, `--lr`, and scheduler parameters if training diverges.
- **Augmentations**: Tune `augmentations_prob` in the dataset constructor to balance diversity and overfitting. All augmentations are designed to preserve label semantics.
- **Checkpoints**: Saved under `out-checkpoints/<experiment_name>/checkpoint_[t|v]_<index>.pth`. Use `torch.load` to inspect or resume training.
- **Plots**: Loss/LR plots are saved automatically when `--plot_stats` or `--plot_lr` are true; disable when running on headless servers without matplotlib backend configuration.

For detailed methodology and experimental results, refer to the accompanying WACV 2022 paper linked in `README.md`.
