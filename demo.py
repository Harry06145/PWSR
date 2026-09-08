"""Real-image 2x super-resolution with a trained PWSR model.

Each input image is treated as a low-resolution grayscale image and is
upscaled by 2x. The physical acquisition conditions (wavelength and NA)
must be supplied; defaults fall back to ``config.yaml -> test.fixed``.

Usage:
    python demo.py --checkpoint checkpoints/best.pth \
        --input /path/to/image.png --output_dir demo_output \
        --lambda 1500 --NA 0.09

    # Process a whole directory
    python demo.py --checkpoint checkpoints/best.pth \
        --input /path/to/images --output_dir demo_output
"""

import argparse
import os
from glob import glob

import cv2
import torch
import yaml

from models.pwsr_arch import PWSR
from psf_utils import normalize_guided_condition


EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def build_model(cfg: dict, checkpoint: str, device: torch.device):
    """Build the PWSR model and load the checkpoint state dict."""
    model_cfg = cfg['model']
    cond_cfg = cfg['condition']
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

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def pad_to_even(img: torch.Tensor):
    """Reflect-pad the right/bottom so that both spatial dims are even."""
    _, h, w = img.shape
    pad_r = h % 2
    pad_b = w % 2
    if pad_r or pad_b:
        img = torch.nn.functional.pad(
            img, (0, pad_b, 0, pad_r), mode='reflect')
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--input', type=str, required=True,
                    help='Input image file or directory of images.')
    ap.add_argument('--output_dir', type=str, default='demo_output')
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--lambda', type=float, default=None, dest='lambda_nm',
                    help='Acquisition wavelength in nm (default: config fixed).')
    ap.add_argument('--NA', type=float, default=None,
                    help='Acquisition numerical aperture (default: config fixed).')
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    lam = args.lambda_nm
    na = args.NA
    if lam is None or na is None:
        fixed = cfg['test']['fixed']
        lam = fixed['lambda'] if lam is None else lam
        na = fixed['NA'] if na is None else na

    device = torch.device(args.device)
    model = build_model(cfg, args.checkpoint, device)
    cond = torch.from_numpy(
        normalize_guided_condition(lam, na, cfg)
    ).float().unsqueeze(0).to(device)   # (1, 2)

    if os.path.isdir(args.input):
        paths = sorted(
            p for p in glob(os.path.join(args.input, '*'))
            if p.lower().endswith(EXTS))
    else:
        paths = [args.input]
    if not paths:
        raise FileNotFoundError(f'No supported images found at: {args.input}')

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Conditions: lambda={lam} nm, NA={na}')
    print(f'Found {len(paths)} image(s); saving to {args.output_dir}')

    with torch.no_grad():
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'  skip (unreadable): {p}')
                continue
            h_in, w_in = img.shape
            lq = torch.from_numpy(img.astype('float32') / 255.0) \
                .unsqueeze(0).unsqueeze(0).to(device)      # (1, 1, H, W)
            lq = pad_to_even(lq)
            _, sr = model(lq, cond)
            sr = sr[:, :, :2 * h_in, :2 * w_in]            # undo even-padding
            out = (sr.squeeze().clamp(0, 1).cpu().numpy()
                   * 255.0).round().astype('uint8')
            name = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(args.output_dir, f'{name}_sr2x.png')
            cv2.imwrite(out_path, out)
            print(f'  {name}: {h_in}x{w_in} -> {out.shape[0]}x{out.shape[1]} '
                  f'-> {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()
