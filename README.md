# PWSR: Physical-Aware Parallel Dual-Domain Network for Robust NIR-II Fluorescence Super-Resolution

Official PyTorch implementation of **PWSR**, a parallel dual-domain super-resolution framework for near-infrared second-window (NIR-II) fluorescence imaging.

> **PWSR: Physical-Aware Parallel Dual-Domain Network for Robust NIR-II Fluorescence Super-Resolution**
> Hao Li, Yifan Zhou, Menghan Guan, Jie Tian, Zhenhua Hu
> The code and algorithm are for non-comercial use only.
> Copyright 2026, Institute of Automation, Chinese Academy of Sciences, Beijing, China.



## Overview

PWSR jointly exploits **wavelet-domain** high-frequency texture extraction and **spatial-domain** global context modeling in two parallel branches, avoiding the error propagation and modality-specific overfitting of serial architectures.

Key features:

- **Physics-guided degradation**: the degradation model is driven by the wavelength (λ) and numerical aperture (NA) of the imaging system. A random unstructured kernel is reshaped so that its radial profile follows the expected physical blur (`σ_px = α·λ/NA`), which keeps the synthetic PSF realistic across both macroscopic and microscopic objectives without collapsing to a point source.
- **Physical condition injection**: (λ, NA) are encoded by a small MLP and spatially tiled into the network as extra channels, letting one model adapt to diverse imaging conditions.
- **Parallel dual-domain architecture**: a wavelet branch (GELU ResBlocks at half resolution) and a spatial branch (CATANet TAB/LRSA blocks at full resolution), fused by a learned convolution head.
- **Multi-view consistency loss**: during training each image is degraded under `k` different PSF conditions; predictions across views are encouraged to be consistent (`k = 3` by default).
- **Lightweight**: ~0.80M parameters, suitable for real-time NIR-II imaging pipelines.

## Architecture

```
Input: lq (B, 1, H, W) + conditions (B, 2)
  |
  +--> Condition MLP -> spatial tile -> concat -> x (B, 1+cond_dim, H, W)
  |
  +--> Wavelet Branch (H/2 x W/2):
  |      DWT -> Conv -> ResBlocks x8 -> PixelShuffle -> IWT -> sr_wavelet
  |
  +--> Spatial Branch (H x W):
  |      Conv -> TAB+LRSA blocks x8 -> PixelShuffle -> sr_spatial
  |
  v
Fusion: learned conv fusion of sr_wavelet and sr_spatial (2x upsampled)
```

## Installation

```bash
conda create -n pwsr python=3.10
conda activate pwsr

# PyTorch >= 2.1 (required for F.scaled_dot_product_attention)
pip install torch>=2.1 torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Dataset Preparation

Only ground-truth (high-resolution) images are required — the physics-guided degradation is applied on the fly during training. Organise the dataset as follows:

```
datasets/
└── WFI_B/
    ├── train/       # training GT images (grayscale .png)
    ├── val/         # validation GT images
    └── test/        # test GT images
```

Then point `dataset.dataroot` in `config.yaml` to your dataset directory (e.g. `./datasets/WFI_B`). GT images are center-cropped to `gt_size x gt_size` (default 512×512) during loading; the degradation pipeline (guided PSF convolution → Poisson/Gaussian noise → 2× average pooling) then produces the 256×256 low-resolution input.

### Adapting the physical model to another system

The default config follows the macroscopic WFI system used in the paper (λ ∈ [1500, 1700] nm, NA ∈ [0.03, 0.15], 32×32 PSF kernel, `α ≈ 9.85e-5`). For a different microscope, adjust the ranges in `config.yaml` so that the effective PSF width stays well inside the kernel:

```yaml
psf:
  lambda_min: 1500.0
  lambda_max: 1700.0
  NA_min: 0.3       # e.g. high-NA LSM objective
  NA_max: 0.8
  alpha: 9.846e-4   # sigma_px = alpha * lambda / NA, scaled so it fits the kernel
```

## Training

```bash
# Train with the default config (k = 3 views + consistency loss)
python train.py --config config.yaml

# Multi-GPU
python train.py --config config.yaml --gpus 0,1

# Resume from a checkpoint
python train.py --config config.yaml --resume checkpoints/epoch_40.pth
```

Checkpoints and text logs are written to `checkpoints/` and `logs/` (configurable in `config.yaml`). The best validation model is stored as `checkpoints/best.pth`; early stopping is enabled by default.

### CheckPonits
Checkpoint will be released after the paper is accepted.

## Testing

`test.py` always evaluates with `k = 1` (one PSF condition per image); the multi-view setting is used only for training. Results (HR/LR/SR images, per-image `metrics.csv` and a summary) are saved to `result/<timestamp>/`.

```bash
# Random PSF degradation sampled from the configured ranges
python test.py --checkpoint checkpoints/best.pth

# Fixed PSF parameters (defaults in config.yaml -> test.fixed)
python test.py --checkpoint checkpoints/best.pth --mode fixed
python test.py --checkpoint checkpoints/best.pth --mode fixed --lambda 1600 --NA 0.05

# Reuse the exact LR images / conditions of a previous run
# (fair comparison of several models on identical degradation)
python test.py --checkpoint checkpoints/best.pth \
    --reuse_dir result/20250101_000000

# Direct mode: paired real-microscope GT/LR images (no synthetic degradation)
python test.py --checkpoint checkpoints/best.pth \
    --gt_dir /path/to/GT --lr_dir /path/to/LR
python test.py --checkpoint checkpoints/best.pth \
    --gt_dir /path/to/GT --lr_dir /path/to/LR --lambda 1500 --NA 0.45
```

PSNR/SSIM are computed on border-shaved images by default (`metrics.crop_border: 4`); set it to `0` for full-image metrics.

## Inference on real images

To super-resolve arbitrary grayscale images (e.g. microscope captures that are not part of a paired benchmark), use `demo.py`. The input is treated as the low-resolution image and is upscaled 2×; supply the physical conditions of the acquisition:

```bash
python demo.py --checkpoint checkpoints/best.pth \
    --input /path/to/image.png --output_dir demo_output \
    --lambda 1500 --NA 0.09

# Process a whole directory
python demo.py --checkpoint checkpoints/best.pth \
    --input /path/to/images/ --output_dir demo_output --lambda 1500 --NA 0.09
```

If the input is a native full-resolution image, first downscale it by 2 (e.g. with area interpolation) so the network receives the same LR domain it was trained on.

## Repository Structure

```
PWSR/
├── config.yaml          # main configuration (k = 3, WFI physical model)
├── train.py             # training with consistency + wavelet-domain losses
├── test.py              # evaluation (random / fixed / reuse / direct modes)
├── demo.py              # single-image / folder inference
├── dataset.py           # on-the-fly physics-degradation dataset
├── psf_utils.py         # guided PSF generation and degradation pipeline
├── models/
│   ├── pwsr_arch.py     # PWSR network (wavelet + spatial branches)
│   └── __init__.py
├── catanet/             # minimal CATANet components used by the spatial branch
├── requirements.txt
├── LICENSE              # MIT license
└── licenses/            # third-party licenses (Apache-2.0 for CATANet/BasicSR)
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2026pwsr,
  title={PWSR: Physical-Aware Parallel Dual-Domain Network for
         Robust NIR-II Fluorescence Super-Resolution},
  author={Li, Hao and Zhou, Yifan and Guan, Menghan and
          Tian, Jie and Hu, Zhenhua},
  journal={Not Published},
  year={2026}
}
```

## Acknowledgements

The spatial-domain attention blocks build upon [CATANet](https://github.com/EquationWalker/CATANet) (Liu et al., CVPR 2025, Apache-2.0). The `trunc_normal_` utility is extracted from [BasicSR](https://github.com/XPixelGroup/BasicSR) (Apache-2.0). The wavelet transform uses [ptwt](https://github.com/v0lta/PyTorch-Wavelet-Toolbox). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

## License

This project is released under the MIT License. Third-party components are governed by their own licenses; see `licenses/`.
