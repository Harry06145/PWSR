"""
Physics-guided PSF generation and degradation pipeline.

The imaging model is guided by the physical parameters of the system:

  * wavelength ``lambda`` (nm),
  * numerical aperture ``NA``.

Given a random unstructured kernel, the PSF is reshaped so that its
radial profile follows the expected physical blur (Gaussian envelope
with ``sigma = alpha * lambda / NA`` by default, or an Airy profile),
which avoids the degenerate point-source collapse that a naive Airy
model produces at large NA on small kernels.

The degradation applied during training is:

  1. convolution with the guided PSF kernel,
  2. Poisson (photon shot) noise + Gaussian (readout) noise,
  3. 2x average pooling (pixel binning).
"""

import numpy as np
from scipy.special import j1
from scipy.ndimage import gaussian_filter
from scipy.optimize import isotonic_regression
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# Physical parameter sampling
# ============================================================================

def _guided_grids(config: dict):
    """Build the uniform NA/lambda grids defined in ``config['psf']``.

    Grids are quantised so that different runs share the same set of
    discrete conditions (e.g. NA step 0.01, lambda step 1 nm).
    """
    psf_cfg = config['psf']
    na_step = psf_cfg.get('na_step', 0.01)
    lam_step = int(psf_cfg.get('lam_step', 1))
    na_grid = np.round(
        np.arange(psf_cfg['NA_min'], psf_cfg['NA_max'] + 1e-9, na_step), 2)
    lam_grid = np.arange(int(psf_cfg['lambda_min']),
                         int(psf_cfg['lambda_max']) + 1,
                         lam_step).astype(float)
    return na_grid, lam_grid


def sample_guided_conditions(config: dict,
                             rng: Optional[np.random.RandomState] = None
                             ) -> Tuple[float, float]:
    """Uniformly sample one independent (lambda, NA) pair."""
    rng = np.random if rng is None else rng
    na_grid, lam_grid = _guided_grids(config)
    return float(rng.choice(lam_grid)), float(rng.choice(na_grid))


def normalize_guided_condition(lam: float, na: float, config: dict) -> np.ndarray:
    """Normalise (lambda, NA) to [0, 1] for the condition MLP input."""
    psf_cfg = config['psf']
    return np.array([
        (lam - psf_cfg['lambda_min']) / (psf_cfg['lambda_max'] - psf_cfg['lambda_min']),
        (na - psf_cfg['NA_min']) / (psf_cfg['NA_max'] - psf_cfg['NA_min']),
    ], dtype=np.float32)


# ============================================================================
# Random unstructured kernel helpers
# ============================================================================

def _random_kernel(size: int, seed=None) -> np.ndarray:
    """Random unstructured kernel with positive and negative values."""
    r = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)
    K = r.randn(size, size) * 0.8 + r.rand(size, size) * 0.2
    K -= K.mean()
    return K


def _radial_profile(K: np.ndarray, rbins: np.ndarray, max_r: int) -> np.ndarray:
    """Azimuthally averaged radial profile of a 2D kernel."""
    return np.array([K[rbins == rr].mean() if (rbins == rr).any() else 0.0
                     for rr in range(max_r + 1)])


def _airy_1d(r_px: np.ndarray, lam: float, na: float, alpha: float) -> np.ndarray:
    """1D Airy profile, mapped to pixels through the calibration ``alpha``."""
    x = 2 * np.pi * na * r_px * (0.21 / alpha) / lam
    out = np.ones_like(r_px, dtype=float)
    nz = x > 0
    out[nz] = (2 * j1(x[nz]) / x[nz]) ** 2
    return out


# ============================================================================
# Guided PSF shaping
# ============================================================================

def _shape_into_psf_gaussian(K: np.ndarray, lam: float, na: float,
                             alpha: float, sigma_smooth: float = 1.0):
    """Gaussian-guided shaping.

    Pipeline: non-negative projection -> azimuthal averaging ->
    monotone (isotonic) regression -> Gaussian envelope with
    ``sigma = alpha * lambda / NA`` -> smoothing -> normalisation.
    """
    H, W = K.shape
    cy, cx = (H - 1) / 2, (W - 1) / 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rbins = np.round(r).astype(int)
    max_r = int(np.ceil(r.max()))

    K_pos = np.maximum(K, 0.0)
    radial = _radial_profile(K_pos, rbins, max_r)
    if len(radial) > 1 and radial[0] == 0.0:
        radial[0] = radial[1]
    mono = -isotonic_regression(-radial).x

    sigma_env = alpha * lam / na                        # physical guidance: lambda, NA
    sigma_smooth = min(sigma_smooth, 0.3 * sigma_env)
    rr = np.arange(max_r + 1)
    env = np.exp(-(rr ** 2) / (2 * sigma_env ** 2))
    shaped = mono * env
    smooth = gaussian_filter(shaped[np.clip(rbins, 0, max_r)], sigma_smooth)
    K_psf = smooth / smooth.sum()
    return K_pos, radial, mono, K_psf, sigma_env


def _shape_into_psf_airy(K: np.ndarray, lam: float, na: float,
                         alpha: float, eps: float = 0.15,
                         sigma_smooth: float = 0.5):
    """Airy-guided shaping: least-squares fit of the azimuthal profile to a
    real Airy profile, keeping part of the diffraction rings."""
    H, W = K.shape
    cy, cx = (H - 1) / 2, (W - 1) / 2
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rbins = np.round(r).astype(int)
    max_r = int(np.ceil(r.max()))

    K_pos = np.maximum(K, 0.0)
    radial = _radial_profile(K_pos, rbins, max_r)
    if len(radial) > 1 and radial[0] == 0.0:
        radial[0] = radial[1]

    rr = np.arange(max_r + 1)
    A = _airy_1d(rr, lam, na, alpha)
    w = rr + 1.0                                       # azimuthal area weight ~ r
    c = max(0.0, float(np.sum(radial * A * w) / np.sum(A * A * w)))
    radial_shaped = c * A + eps * np.maximum(radial - c * A, 0.0)

    shaped = radial_shaped[np.clip(rbins, 0, max_r)]
    smooth = gaussian_filter(shaped, sigma_smooth)
    K_psf = smooth / smooth.sum()
    return K_pos, radial, radial_shaped, K_psf, c


def generate_guided_psf(lam: float, na: float, config: dict,
                        mode: Optional[str] = None, seed=None) -> np.ndarray:
    """Generate a physics-guided PSF kernel (normalised, sum = 1).

    Args:
        lam:    Wavelength in nm.
        na:     Numerical aperture.
        config: Configuration dict (uses ``psf.kernel_size`` and ``psf.alpha``).
        mode:   ``"gaussian"`` (default, from config) or ``"airy"``.
        seed:   Seed / RandomState for the initial random kernel.
    """
    psf_cfg = config['psf']
    mode = mode or psf_cfg.get('psf_mode', 'gaussian')
    alpha = psf_cfg.get('alpha', 0.0002)
    ks = psf_cfg['kernel_size']
    K = _random_kernel(ks, seed)
    if mode == 'airy':
        return _shape_into_psf_airy(K, lam, na, alpha)[3]
    return _shape_into_psf_gaussian(K, lam, na, alpha)[3]


# ============================================================================
# Degradation
# ============================================================================

def degrade_image_torch_guided(img_gt: torch.Tensor,
                               psf: torch.Tensor,
                               poisson_scale: float = 0.02,
                               gauss_sigma: float = 0.01) -> torch.Tensor:
    """Degrade a ground-truth image: convolution -> noise -> 2x pooling.

    Args:
        img_gt:        Ground truth tensor ``(1, H, W)`` in [0, 1].
        psf:           PSF kernel tensor ``(ks, ks)`` with sum 1.
        poisson_scale: Mixing weight of Poisson (shot) noise.
        gauss_sigma:   Std of Gaussian (readout) noise.

    Returns:
        Degraded image ``(1, H//2, W//2)`` in [0, 1].
    """
    _, H, W = img_gt.shape
    ks = psf.shape[0]
    pad = ks // 2
    img_4d = img_gt.unsqueeze(0)                       # (1, 1, H, W)
    img_blur = F.conv2d(F.pad(img_4d, (pad, pad, pad, pad), mode='reflect'),
                        psf.view(1, 1, ks, ks))

    # Poisson (photon counting) noise
    vals = 255.0
    noisy = torch.poisson(img_blur.clamp(0, 1) * vals) / vals
    img_noisy = img_blur + poisson_scale * (noisy - img_blur)
    # Gaussian (readout) noise
    if gauss_sigma > 0:
        img_noisy = img_noisy + gauss_sigma * torch.randn_like(img_blur)
    img_noisy = img_noisy.clamp(0, 1)

    # 2x average pooling (simulated pixel binning)
    return F.avg_pool2d(img_noisy, kernel_size=2, stride=2).squeeze(0)
