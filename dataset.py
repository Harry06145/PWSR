"""
PyTorch dataset for physics-based super-resolution.

For each ground-truth image we sample ``k`` different PSF conditions
(multi-view contrastive learning), apply the physics-guided degradation
pipeline, and return the degraded low-resolution images together with the
normalised condition vectors ``(lambda, NA)``.

Only ground-truth images need to be stored on disk; the degradation is
applied on the fly. GT images should be square ``gt_size x gt_size`` crops
(the loader center-crops larger images).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List
import cv2

from psf_utils import (
    sample_guided_conditions,
    normalize_guided_condition,
    generate_guided_psf,
    degrade_image_torch_guided,
)


class MiceBDataset(Dataset):
    """
    Grayscale image dataset with on-the-fly physics degradation.

    Training always samples random PSF parameters (multi-view contrastive
    setting). Validation/test additionally support the ``"random"``,
    ``"fixed"``, ``"reuse"`` and ``"direct"`` modes configured in
    ``config['test']['psf_mode']``.

    Each ``__getitem__`` returns:

    * ``lq``             -- ``(k, 1, H_lq, W_lq)`` degraded views,
    * ``gt``             -- ``(1, H_gt, W_gt)`` ground truth,
    * ``conditions``     -- ``(k, 2)`` normalised (lambda, NA),
    * ``conditions_raw`` -- list of ``k`` raw ``(lambda, NA)`` tuples,
    * ``img_name``       -- file name without extension.
    """

    def __init__(self,
                 config: dict,
                 split: str = 'train',
                 seed: int = 42):
        """
        Args:
            config: Configuration dict loaded from config.yaml.
            split:  One of {'train', 'val', 'test'}.
            seed:   Random seed for reproducibility.
        """
        self.config = config
        self.split = split
        self.k = config['degradation']['num_conditions']
        self.ks = config['psf']['kernel_size']
        self.mode = config['psf'].get('psf_mode', 'gaussian')
        self.gt_size = config['dataset'].get('gt_size', 512)
        self.lq_size = config['dataset'].get('lq_size', 256)
        self.rng = np.random.RandomState(seed)

        # Determine PSF sampling mode
        if split == 'train':
            self.psf_mode = 'random'  # training is always random
        else:
            self.psf_mode = config.get('test', {}).get('psf_mode', 'random')

        # Paths (skip dataroot loading for direct mode — handled in _init_direct_mode)
        if self.psf_mode != 'direct':
            self.dataroot = os.path.join(config['dataset']['dataroot'], split)
            self.img_names = sorted(
                [f for f in os.listdir(self.dataroot) if f.lower().endswith('.png')]
            )
            if len(self.img_names) == 0:
                raise FileNotFoundError(f"No PNG images found in {self.dataroot}")
            # Optional cap on the number of images (0 = all)
            max_images = config.get('dataset', {}).get('max_images', 0)
            if max_images and len(self.img_names) > max_images:
                self.img_names = self.img_names[:max_images]
                print(f"[{split}] Capped to first {len(self.img_names)} images "
                      f"(max_images={max_images})")
            print(f"[{split}] Loaded {len(self.img_names)} images from {self.dataroot}")

        # Mode-specific setup (reuse needs self.img_names, so run after loading)
        if split == 'train':
            pass  # training is always random on-the-fly degradation
        elif self.psf_mode == 'fixed':
            fixed_cfg = config.get('test', {}).get('fixed', {})
            self.fixed_lam = fixed_cfg.get('lambda', 1600.0)
            self.fixed_na = fixed_cfg.get('NA', 0.09)
            self.fixed_poisson = fixed_cfg.get('poisson_scale', 0.01)
            self.fixed_gauss = fixed_cfg.get('gauss_sigma', 0.005)
            print(f"[{split}] Using FIXED PSF: λ={self.fixed_lam}nm, "
                  f"NA={self.fixed_na}, poisson_scale={self.fixed_poisson}, "
                  f"gauss_sigma={self.fixed_gauss}")
        elif self.psf_mode == 'reuse':
            self.reuse_dir = config.get('test', {}).get('reuse_dir', '')
            if not self.reuse_dir:
                raise ValueError("psf_mode='reuse' requires test.reuse_dir in config")
            self._init_reuse_mode(split)
        elif self.psf_mode == 'direct':
            self.gt_dir = config.get('test', {}).get('gt_dir', '')
            self.lr_dir = config.get('test', {}).get('lr_dir', '')
            if not self.gt_dir or not self.lr_dir:
                raise ValueError("psf_mode='direct' requires test.gt_dir and test.lr_dir in config")
            self._init_direct_mode(split)
        else:
            print(f"[{split}] Using RANDOM PSF sampling")

    def _init_reuse_mode(self, split: str):
        """
        Initialise reuse mode: read pre-generated LR images and PSF conditions
        from a previous run's result directory.

        Expects:
            reuse_dir/
                LR/          — pre-degraded low-res images (*.png)
                metrics.csv  — per-image PSF parameters (lambda_nm, NA)
        """
        import csv
        metrics_path = os.path.join(self.reuse_dir, 'metrics.csv')
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"metrics.csv not found in reuse_dir: {self.reuse_dir}. "
                f"Make sure test.py has been run first to generate the dataset."
            )

        self.reuse_params = {}
        with open(metrics_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.reuse_params[row['img_name']] = {
                    'lambda': float(row['lambda_nm']),
                    'NA': float(row['NA']),
                }

        # Keep only images that exist in both the split dir and the reuse params
        before = len(self.img_names)
        self.img_names = [n for n in self.img_names
                          if os.path.splitext(n)[0] in self.reuse_params]
        dropped = before - len(self.img_names)
        if dropped > 0:
            print(f"[{split}] Reuse mode: dropped {dropped} images not found in reuse_dir "
                  f"(kept {len(self.img_names)})")
        print(f"[{split}] Reuse mode: {len(self.img_names)} images from {self.reuse_dir}")

    def _init_direct_mode(self, split: str):
        """
        Initialise direct mode: load GT and LR images from user-specified directories.

        GT images are loaded from gt_dir, LR images from lr_dir.
        Images are matched by filename (intersection of both directories).
        PSF conditions use user-specified values or mid-range defaults from config.
        """
        gt_files = set(f for f in os.listdir(self.gt_dir) if f.lower().endswith('.png'))
        lr_files = set(f for f in os.listdir(self.lr_dir) if f.lower().endswith('.png'))

        if len(gt_files) == 0:
            raise FileNotFoundError(f"No PNG images found in GT dir: {self.gt_dir}")
        if len(lr_files) == 0:
            raise FileNotFoundError(f"No PNG images found in LR dir: {self.lr_dir}")

        # Use intersection — only images present in both directories
        common = sorted(gt_files & lr_files)
        gt_only = len(gt_files) - len(common)
        lr_only = len(lr_files) - len(common)

        if len(common) == 0:
            raise FileNotFoundError(
                f"No common images between GT dir ({len(gt_files)} images) "
                f"and LR dir ({len(lr_files)} images). "
                f"Make sure filenames match exactly."
            )

        self.img_names = common

        # Read user-specified PSF conditions, or use mid-range defaults from config
        direct_cfg = self.config.get('test', {}).get('direct', {})
        psf_cfg = self.config['psf']
        self.direct_lam = direct_cfg.get('lambda',
            (psf_cfg['lambda_min'] + psf_cfg['lambda_max']) / 2)
        self.direct_na = direct_cfg.get('NA',
            (psf_cfg['NA_min'] + psf_cfg['NA_max']) / 2)

        print(f"[{split}] Direct mode: {len(common)} matched images "
              f"(GT-only: {gt_only}, LR-only: {lr_only})")
        print(f"[{split}]   GT dir:  {self.gt_dir}")
        print(f"[{split}]   LR dir:  {self.lr_dir}")
        print(f"[{split}]   Conditions: λ={self.direct_lam:.1f}nm, "
              f"NA={self.direct_na:.4f}")

    def __len__(self) -> int:
        return len(self.img_names)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load a grayscale PNG as float32 tensor (1, H, W) in [0, 1]."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to load {path}")
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

    def _crop_center(self, img: torch.Tensor, size: int) -> torch.Tensor:
        """Center-crop to ``(1, size, size)`` for consistent batch sizes."""
        _, H, W = img.shape
        if H == size and W == size:
            return img
        y0 = max(0, (H - size) // 2)
        x0 = max(0, (W - size) // 2)
        return img[:, y0:y0 + size, x0:x0 + size]

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random horizontal flip and/or 90° rotation (training only)."""
        if self.split != 'train':
            return img
        cfg = self.config['dataset']
        if cfg.get('use_hflip', False) and self.rng.rand() > 0.5:
            img = torch.flip(img, dims=[-1])
        if cfg.get('use_rot', False) and self.rng.rand() > 0.5:
            k = self.rng.randint(1, 4)
            img = torch.rot90(img, k, dims=[-2, -1])
        return img

    def _get_psf_params(self):
        """
        Sample PSF parameters according to the current mode.

        Returns:
            (lam, na, poisson_scale, gauss_sigma)
        """
        if self.psf_mode == 'fixed':
            return (self.fixed_lam, self.fixed_na,
                    self.fixed_poisson, self.fixed_gauss)
        else:
            # Random mode: sample lambda and NA independently and uniformly
            lam, na = sample_guided_conditions(self.config, self.rng)
            poisson_scale = self.rng.uniform(
                self.config['degradation']['poisson_scale_min'],
                self.config['degradation']['poisson_scale_max']
            )
            gauss_sigma = self.rng.uniform(
                self.config['degradation']['gauss_sigma_min'],
                self.config['degradation']['gauss_sigma_max']
            )
            return lam, na, poisson_scale, gauss_sigma

    def __getitem__(self, idx: int) -> Dict:
        img_name = self.img_names[idx]

        # --- Direct mode: load GT and LR from user-specified directories ---
        if self.psf_mode == 'direct':
            gt_path = os.path.join(self.gt_dir, img_name)
            lr_path = os.path.join(self.lr_dir, img_name)

            img_gt = self._load_image(gt_path)  # (1, H, W)
            img_lq = self._load_image(lr_path)  # (1, H, W)
            img_gt = self._crop_center(img_gt, self.gt_size)
            img_lq = self._crop_center(img_lq, self.lq_size)

            cond_norm = normalize_guided_condition(
                self.direct_lam, self.direct_na, self.config
            )
            cond_raw = (self.direct_lam, self.direct_na)

            return {
                'lq': img_lq.unsqueeze(0),                               # (1, 1, H, W)  k=1
                'gt': img_gt,                                             # (1, H, W)
                'conditions': torch.from_numpy(cond_norm).unsqueeze(0),   # (1, 2)
                'conditions_raw': [cond_raw],                             # list of 1 tuple
                'img_name': os.path.splitext(img_name)[0],
            }

        # --- All other modes: load GT from dataroot ---
        img_path = os.path.join(self.dataroot, img_name)
        img_gt = self._load_image(img_path)  # (1, 512, 512)
        img_gt = self._crop_center(img_gt, self.gt_size)

        # --- Reuse mode: load pre-degraded LR, skip on-the-fly degradation ---
        if self.psf_mode == 'reuse':
            name_no_ext = os.path.splitext(img_name)[0]
            lr_path = os.path.join(self.reuse_dir, 'LR', f'{name_no_ext}.png')
            if not os.path.exists(lr_path):
                raise FileNotFoundError(
                    f"LR image not found: {lr_path}. "
                    f"Make sure test.py generated LR images in reuse_dir."
                )
            img_lq = self._load_image(lr_path)  # (1, 256, 256)
            img_lq = self._crop_center(img_lq, self.lq_size)

            params = self.reuse_params[name_no_ext]
            cond_norm = normalize_guided_condition(
                params['lambda'], params['NA'], self.config
            )
            cond_raw = (params['lambda'], params['NA'])

            return {
                'lq': img_lq.unsqueeze(0),                          # (1, 1, 256, 256)  k=1
                'gt': img_gt,                                        # (1, 512, 512)
                'conditions': torch.from_numpy(cond_norm).unsqueeze(0),  # (1, 2)
                'conditions_raw': [cond_raw],                        # list of 1 tuple
                'img_name': name_no_ext,
            }

        # --- Random / Fixed mode: on-the-fly degradation ---
        # Augmentation (training only, applied to GT before degradation)
        img_gt = self._augment(img_gt)
        img_gt = self._crop_center(img_gt, self.gt_size)

        # Sample k PSF conditions and degrade (guided by lambda/NA only)
        lq_list = []
        cond_norm_list = []
        cond_raw_list = []

        for _ in range(self.k):
            # In fixed mode, all k conditions use the same fixed PSF;
            # in random mode, each gets a new random sample.
            lam, na, poisson_scale, gauss_sigma = self._get_psf_params()
            cond_raw_list.append((lam, na))

            # Normalize condition
            cond_norm = normalize_guided_condition(lam, na, self.config)
            cond_norm_list.append(cond_norm)

            # Guided PSF & degradation: conv -> Poisson+Gaussian -> avgpool 2x
            psf_np = generate_guided_psf(lam, na, self.config,
                                         mode=self.mode, seed=self.rng)
            psf = torch.from_numpy(psf_np.astype(np.float32))
            img_lq = degrade_image_torch_guided(img_gt, psf,
                                                poisson_scale, gauss_sigma)
            lq_list.append(img_lq)

        return {
            'lq': torch.stack(lq_list),                          # (k, 1, 256, 256)
            'gt': img_gt,                                         # (1, 512, 512)
            'conditions': torch.from_numpy(np.stack(cond_norm_list)),  # (k, 2)
            'conditions_raw': cond_raw_list,                      # list of k tuples
            'img_name': os.path.splitext(img_name)[0],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function — stacks all keys.

    After collation:
        lq:         (B, k, 1, 256, 256)
        gt:         (B, 1, 512, 512)
        conditions: (B, k, 2)
    """
    out = {}
    for key in ['lq', 'gt', 'conditions']:
        out[key] = torch.stack([item[key] for item in batch])
    out['conditions_raw'] = [item['conditions_raw'] for item in batch]
    out['img_name'] = [item['img_name'] for item in batch]
    return out


# ============================================================================
# Quick test
# ============================================================================
if __name__ == '__main__':
    import yaml

    with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    ds = MiceBDataset(config, split='train')
    print(f"Dataset size: {len(ds)}")

    sample = ds[0]
    print(f"  lq shape:         {sample['lq'].shape}")
    print(f"  gt shape:          {sample['gt'].shape}")
    print(f"  conditions shape:  {sample['conditions'].shape}")
    print(f"  conditions (raw):  {sample['conditions_raw']}")
    print(f"  lq range:          [{sample['lq'].min():.4f}, {sample['lq'].max():.4f}]")
    print(f"  gt range:          [{sample['gt'].min():.4f}, {sample['gt'].max():.4f}]")
