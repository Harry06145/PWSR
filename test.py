"""
Test / Evaluation script for PWSR.

Testing uses k=1 (single PSF condition) — contrastive multi-condition is
only needed for training. Results (HR/LR/SR images + metrics) are saved
under result/<timestamp>/.

Supports four PSF modes (configurable in config.yaml -> test section):
  - "random": randomly sample PSF parameters from the configured ranges
  - "fixed":  use the exact lambda, NA values specified in config
  - "reuse":  load pre-generated LR images & PSF conditions from a previous
              run (for fair ablation comparison)
  - "direct": load GT and LR images directly from specified directories,
              bypassing PSF degradation (for real microscope data)

Usage:
    # Full model: generate random degradation
    python test.py --checkpoint checkpoints/best.pth

    # Reuse the same degradation as a previous run
    python test.py --checkpoint checkpoints/best.pth --reuse_dir result/20240717_120000

    # Fixed PSF
    python test.py --checkpoint checkpoints/best.pth --mode fixed
    python test.py --checkpoint checkpoints/best.pth --mode fixed --lambda 1600 --NA 0.05

    # Direct: use real microscope GT/LR images (no synthetic degradation)
    python test.py --checkpoint checkpoints/best.pth --gt_dir /path/to/GT --lr_dir /path/to/LR
"""

import os
import csv
import argparse
from datetime import datetime
import yaml
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from dataset import MiceBDataset, collate_fn
from models.pwsr_arch import PWSR


def compute_ssim(img1, img2):
    """Compute SSIM between two single-channel numpy images in [0, 255] uint8."""
    return ssim(img1, img2, data_range=255)


@torch.no_grad()
def test(model, dataloader, device, result_dir, border=0):
    """Run evaluation on the test set (k=1 condition per image).

    Saves HR (ground truth), LR (low-res input), SR (super-resolved output)
    images, and per-image metrics CSV under result_dir.

    Args:
        model:      PWSR model.
        dataloader: Test dataloader.
        device:     torch device.
        result_dir: Directory to save HR/LR/SR/ subdirs and metrics CSV.

    Returns:
        avg_psnr, avg_ssim
    """
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    n_samples = 0

    metrics_records = []

    # Create subdirectories for HR, LR, SR
    hr_dir = os.path.join(result_dir, 'HR')
    lr_dir = os.path.join(result_dir, 'LR')
    sr_dir = os.path.join(result_dir, 'SR')
    for d in [hr_dir, lr_dir, sr_dir]:
        os.makedirs(d, exist_ok=True)

    for batch in tqdm(dataloader, desc='Testing'):
        lq = batch['lq'].to(device)              # (B, 1, 1, 256, 256)
        gt = batch['gt'].to(device)              # (B, 1, 512, 512)
        conditions = batch['conditions'].to(device)  # (B, 1, 2) — (λ, NA)
        img_names = batch['img_name']

        B = lq.size(0)

        for b in range(B):
            # k=1: single forward pass per image
            sr = model(lq[b:b+1, 0], conditions[b:b+1, 0])[1]  # (1, 1, 512, 512)

            # Metrics use border-shave crops (SR convention); full images
            # are still saved to disk
            if border and border > 0:
                sr_met = sr[0, :, border:-border, border:-border]
                gt_met = gt[b, :, border:-border, border:-border]
            else:
                sr_met, gt_met = sr[0], gt[b]
            mse = torch.nn.functional.mse_loss(sr_met, gt_met)
            psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse))
            total_psnr += psnr.item()
            n_samples += 1

            # SSIM
            sr_np = (sr_met[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            gt_np = (gt_met[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            ssim_val = compute_ssim(sr_np, gt_np)
            total_ssim += ssim_val

            # Record metrics
            cond_raw = batch['conditions_raw'][b][0]
            metrics_records.append({
                'img_name': img_names[b],
                'lambda_nm': cond_raw[0],
                'NA': cond_raw[1],
                'PSNR': round(psnr.item(), 4),
                'SSIM': round(ssim_val, 6),
            })

            # Save HR (ground truth)
            name = img_names[b]
            gt_img = (gt[b, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(hr_dir, f'{name}.png'), gt_img)

            # Save LR (low-res input)
            lr_img = (lq[b, 0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(lr_dir, f'{name}.png'), lr_img)

            # Save SR (super-resolved output)
            sr_img = (sr[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(sr_dir, f'{name}.png'), sr_img)

    avg_psnr = total_psnr / n_samples if n_samples > 0 else 0.0
    avg_ssim = total_ssim / n_samples if n_samples > 0 else 0.0

    # --- Save per-image metrics CSV ---
    if metrics_records:
        csv_path = os.path.join(result_dir, 'metrics.csv')
        fieldnames = ['img_name', 'lambda_nm', 'NA', 'PSNR', 'SSIM']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_records)
        print(f"\nPer-image metrics saved to {csv_path} ({len(metrics_records)} images)")

        # --- Save summary CSV ---
        summary_path = os.path.join(result_dir, 'summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['num_images', 'avg_PSNR', 'avg_SSIM'])
            writer.writeheader()
            writer.writerow({
                'num_images': len(metrics_records),
                'avg_PSNR': round(avg_psnr, 4),
                'avg_SSIM': round(avg_ssim, 6),
            })
        print(f"Summary saved to {summary_path}")

    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint.')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['random', 'fixed'],
                        help='PSF mode: random (sample from ranges) or fixed (use specified values).')
    parser.add_argument('--lambda', type=float, default=None, dest='lambda_nm',
                        help='Wavelength in nm (requires --mode fixed).')
    parser.add_argument('--NA', type=float, default=None,
                        help='Numerical aperture (requires --mode fixed).')
    parser.add_argument('--poisson_scale', type=float, default=None,
                        help='Poisson noise scale (requires --mode fixed).')
    parser.add_argument('--reuse_dir', type=str, default=None,
                        help='Path to a previous result directory (containing '
                             'LR/ and metrics.csv). When set, PSF mode is forced to '
                             '"reuse".')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Path to directory containing GT images. '
                             'When set together with --lr_dir, bypasses PSF degradation.')
    parser.add_argument('--lr_dir', type=str, default=None,
                        help='Path to directory containing LR images. '
                             'Must be used together with --gt_dir.')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use.')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")

    # ---- Testing only needs k=1 ----
    config['degradation']['num_conditions'] = 1

    # Override PSF mode and fixed parameters
    if args.gt_dir and args.lr_dir:
        if not os.path.isdir(args.gt_dir):
            raise FileNotFoundError(f"GT directory not found: {args.gt_dir}")
        if not os.path.isdir(args.lr_dir):
            raise FileNotFoundError(f"LR directory not found: {args.lr_dir}")
        config.setdefault('test', {})['psf_mode'] = 'direct'
        config.setdefault('test', {})['gt_dir'] = args.gt_dir
        config.setdefault('test', {})['lr_dir'] = args.lr_dir
        direct_params = config.setdefault('test', {}).setdefault('direct', {})
        if args.lambda_nm is not None:
            direct_params['lambda'] = args.lambda_nm
        if args.NA is not None:
            direct_params['NA'] = args.NA
    elif args.gt_dir or args.lr_dir:
        raise ValueError('--gt_dir and --lr_dir must be used together (both are required)')
    elif args.reuse_dir:
        config.setdefault('test', {})['psf_mode'] = 'reuse'
        config.setdefault('test', {})['reuse_dir'] = args.reuse_dir
    elif args.mode:
        config.setdefault('test', {})['psf_mode'] = args.mode

    if args.lambda_nm is not None or args.NA is not None:
        current_mode = config.get('test', {}).get('psf_mode', 'random')
        if current_mode == 'direct':
            direct_params = config.setdefault('test', {}).setdefault('direct', {})
            if args.lambda_nm is not None:
                direct_params['lambda'] = args.lambda_nm
            if args.NA is not None:
                direct_params['NA'] = args.NA
        else:
            config.setdefault('test', {})['psf_mode'] = 'fixed'
            fixed = config.setdefault('test', {}).setdefault('fixed', {})
            if args.lambda_nm is not None:
                fixed['lambda'] = args.lambda_nm
            if args.NA is not None:
                fixed['NA'] = args.NA
            if args.poisson_scale is not None:
                fixed['poisson_scale'] = args.poisson_scale

    test_mode = config.get('test', {}).get('psf_mode', 'random')
    if test_mode == 'direct':
        gt_dir = config.get('test', {}).get('gt_dir', '')
        lr_dir = config.get('test', {}).get('lr_dir', '')
        direct_params = config.get('test', {}).get('direct', {})
        lam = direct_params.get('lambda', 'default')
        na = direct_params.get('NA', 'default')
        print(f"Test mode: DIRECT — GT from {gt_dir}, LR from {lr_dir}")
        print(f"  Conditions: lambda={lam}nm, NA={na}")
    elif test_mode == 'reuse':
        reuse_dir = config.get('test', {}).get('reuse_dir', '')
        print(f"Test mode: REUSE — loading LR & conditions from {reuse_dir}")
    elif test_mode == 'fixed':
        fixed = config.get('test', {}).get('fixed', {})
        print(f"Test mode: FIXED — lambda={fixed.get('lambda')}nm, "
              f"NA={fixed.get('NA')}, "
              f"poisson={fixed.get('poisson_scale', 0.01)}")
    else:
        print("Test mode: RANDOM")

    # Device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset (k=1 after config override above)
    test_ds = MiceBDataset(config, split=args.split)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )

    # Model
    model_cfg = config['model']
    cond_cfg = config['condition']
    model = PWSR(
        cond_dim=cond_cfg['cond_dim'],
        cond_input_dim=cond_cfg.get('input_dim', 2),
        hidden_dims=cond_cfg['mlp_hidden'],
        in_chans=model_cfg['in_chans'],
        upscale=model_cfg['upscale'],
        dim=model_cfg.get('dim', 40),
        wavelet=model_cfg.get('wavelet', 'haar'),
        wavelet_blocks=model_cfg.get('wavelet_blocks', 8),
        block_num=model_cfg.get('block_num', 8),
        qk_dim=model_cfg.get('qk_dim', 36),
        mlp_dim=model_cfg.get('mlp_dim', 96),
        heads=model_cfg.get('heads', 4),
        patch_size=model_cfg.get('patch_size'),
        n_iters=model_cfg.get('n_iters'),
        num_tokens=model_cfg.get('num_tokens'),
        group_size=model_cfg.get('group_size'),
        use_wavelet=model_cfg.get('use_wavelet', True),
        use_cata=model_cfg.get('use_cata', True),
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model']
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Create timestamped result directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(os.path.dirname(__file__), 'result', timestamp)
    print(f"Saving results to {result_dir}/")

    # Run test
    border = config.get('metrics', {}).get('crop_border', 0)
    print(f"Metric border crop: {border} px")
    avg_psnr, avg_ssim = test(model, test_loader, device, result_dir, border=border)

    print(f"\n{'='*50}")
    print(f"Test Results ({args.split}, {test_mode}, k=1)")
    print(f"  avg_PSNR: {avg_psnr:.4f} dB")
    print(f"  avg_SSIM: {avg_ssim:.6f}")
    print(f"  saved to: {result_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
