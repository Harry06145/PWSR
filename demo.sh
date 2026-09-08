#!/bin/bash
# ============================================================================
# PWSR: Physical-Aware Parallel Dual-Domain Network for Robust NIR-II
# Fluorescence Super-Resolution
# ============================================================================

# ---------- Training ----------
# Train from scratch
CUDA_VISIBLE_DEVICES=0 python train.py --config config.yaml

# Resume from checkpoint
CUDA_VISIBLE_DEVICES=0 python train.py --config config.yaml --resume checkpoints/epoch_40.pth

# ---------- Testing ----------
# Random PSF mode
CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint checkpoints/best.pth

# Fixed PSF parameters
CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint checkpoints/best.pth --mode fixed

# Manual PSF parameter override
python test.py --checkpoint checkpoints/best.pth --mode fixed \
    --lambda 1700 --NA 0.35

# Reuse degradation from a previous run (same LR for all models)
python test.py --checkpoint checkpoints/best.pth \
    --reuse_dir result/20240717_120000

# Direct mode: use real microscope GT/LR images
python test.py --checkpoint checkpoints/best.pth \
    --gt_dir /path/to/GT --lr_dir /path/to/LR

# Direct mode with specified PSF conditions
python test.py --checkpoint checkpoints/best.pth \
    --gt_dir /path/to/GT --lr_dir /path/to/LR \
    --lambda 1500 --NA 0.45

# Results are saved under result/<timestamp>/ with HR/, LR/, SR/ images and metrics.csv.
