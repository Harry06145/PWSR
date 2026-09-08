"""
Training script for PWSR: Physical-aware Parallel Dual-Domain Network
for Robust NIR-II Fluorescence Super-Resolution.

Physics-aware degradation with contrastive PSF conditions.
"""

import atexit
import json
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from dataset import MiceBDataset, collate_fn
from models.pwsr_arch import PWSR, wt as wavelet_transform


# ============================================================================
# Simple text logger
# ============================================================================

class TextLogger:
    """Plain-text logger — one log file per run."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(log_dir, f'train_{timestamp}.log')
        self.file = open(self.path, 'a', buffering=1)  # line-buffered
        self._write(f"Log started at {timestamp}")

    def _write(self, msg: str):
        ts = datetime.now().strftime('%H:%M:%S')
        self.file.write(f'[{ts}] {msg}\n')

    def log(self, msg: str):
        """Write a plain message (also printed to console)."""
        self._write(msg)

    def log_metrics(self, epoch: int, metrics: dict):
        """Write epoch-level metrics: epoch, loss, l1, cons, lr, psnr, ssim."""
        parts = [f"epoch={epoch}"]
        for k, v in metrics.items():
            if v is not None:
                parts.append(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")
        self._write(' | '.join(parts))

    def close(self):
        self.file.close()


# ============================================================================
# Per-checkpoint-directory training lock
# ============================================================================

def _pid_alive(pid: int) -> bool:
    """Check whether a PID is still alive (without sending any signal)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TrainingLock:
    """Prevent concurrent training processes from writing the same directory.

    Only one process may hold the lock for a given ``save_dir``. If the lock
    file contains a PID that is still alive, training aborts and asks the
    user to point ``logging.save_dir`` to a different directory, so that
    ``best.pth`` / ``epoch_*.pth`` are never overwritten by two runs.
    A stale lock left by a killed process is taken over automatically.
    """

    def __init__(self, save_dir: str, config_name: str):
        self.save_dir = save_dir
        self.config_name = config_name
        self.lock_path = os.path.join(save_dir, ".train.lock")
        self.info_path = os.path.join(save_dir, "run_info.json")
        self.acquired = False

    def acquire(self):
        os.makedirs(self.save_dir, exist_ok=True)
        if os.path.exists(self.lock_path):
            try:
                with open(self.lock_path) as f:
                    info = json.load(f)
            except Exception:
                info = {}
            pid = info.get("pid")
            if pid and _pid_alive(int(pid)) and int(pid) != os.getpid():
                raise RuntimeError(
                    f"Checkpoint directory {self.save_dir} is already in use "
                    f"by another training process:\n"
                    f"  PID={pid}, config={info.get('config')}, "
                    f"start_time={info.get('start_time')}\n"
                    f"To avoid overwriting best.pth / epoch_*.pth, either:\n"
                    f"  1) set logging.save_dir in the config to a separate "
                    f"directory, or\n"
                    f"  2) stop the running process.\n"
                    f"If that process is already dead (stale lock), you can "
                    f"remove {self.lock_path} and retry."
                )

        # Warn if this directory was previously written by another config
        if os.path.exists(self.info_path):
            try:
                with open(self.info_path) as f:
                    prev = json.load(f)
                if prev.get("config") != self.config_name:
                    print(f"WARNING: directory {self.save_dir} was previously "
                          f"written by config '{prev.get('config')}' "
                          f"(PID {prev.get('pid')}, {prev.get('start_time')}); "
                          f"this run uses '{self.config_name}'.")
            except Exception:
                pass

        info = {
            "pid": os.getpid(),
            "config": self.config_name,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.lock_path, "w") as f:
            json.dump(info, f)
        # Run record (overwritten on every start; kept after training so the
        # last run can be traced back to its config and PID)
        with open(self.info_path, "w") as f:
            json.dump(info, f)
        self.acquired = True
        print(f"Acquired training lock: {self.lock_path}")
        print(f"Run info: config={self.config_name}, PID={os.getpid()}")

    def release(self):
        if self.acquired and os.path.exists(self.lock_path):
            try:
                os.remove(self.lock_path)
            except OSError:
                pass
        self.acquired = False


# ============================================================================
# Helper: PSNR
# ============================================================================

def _crop_border(t: torch.Tensor, border: int) -> torch.Tensor:
    """Remove ``border`` pixels on each side (SR border-shave convention)."""
    if border and border > 0:
        return t[..., border:-border, border:-border]
    return t


@torch.no_grad()
def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 border: int = 0) -> float:
    """Compute PSNR between pred and target. Tensors in [0, 1]."""
    pred = _crop_border(pred, border)
    target = _crop_border(target, border)
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return float(20.0 * torch.log10(1.0 / torch.sqrt(mse)))


@torch.no_grad()
def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11, border: int = 0) -> float:
    """Compute SSIM between pred and target (single-channel images).

    Tensors in [0, 1], shape (1, H, W).
    """
    pred = _crop_border(pred, border)
    target = _crop_border(target, border)
    C = pred.shape[0]  # channels = 1

    # Gaussian window
    sigma = 1.5
    gauss = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    gauss = torch.exp(-((gauss - window_size // 2) ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(0) * gauss.unsqueeze(1)  # outer product
    window = window_1d.expand(C, 1, window_size, window_size)

    mu1 = F.conv2d(pred.unsqueeze(0), window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target.unsqueeze(0), window, padding=window_size // 2, groups=C)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d((pred.unsqueeze(0)) ** 2, window,
                         padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d((target.unsqueeze(0)) ** 2, window,
                         padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d((pred.unsqueeze(0) * target.unsqueeze(0)), window,
                       padding=window_size // 2, groups=C) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


# ============================================================================
# Training
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, config, device, epoch):
    """Train for one epoch.

    Each (image, condition) pair is treated as an independent sample —
    forward → backward → step, no gradient accumulation.
    """
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_wavelet = 0.0
    total_cons = 0.0
    n_steps = 0

    loss_cfg = config['train']['loss']
    l1_weight = loss_cfg.get('l1_weight', 1.0)
    cons_weight = loss_cfg.get('consistency_weight', 0.0)
    use_wavelet = (model.module.use_wavelet if hasattr(model, 'module')
                   else model.use_wavelet)
    wavelet_weight = loss_cfg.get('wavelet_weight', 0.1) if use_wavelet else 0.0
    lambda_ll = loss_cfg.get('lambda_ll', 0.01)
    k = config['degradation']['num_conditions']
    wavelet_type = config.get('model', {}).get('wavelet', 'haar')

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        lq = batch['lq'].to(device)              # (B, k, 1, 256, 256)
        gt = batch['gt'].to(device)              # (B, 1, 512, 512)
        conditions = batch['conditions'].to(device)  # (B, k, 2) — (λ, NA)

        # --- Step 1: no_grad forward for all k conditions (for consistency loss) ---
        sr_no_grad = []
        if cons_weight > 0 and k > 1:
            with torch.no_grad():
                for ki in range(k):
                    _, sr_k = model(lq[:, ki], conditions[:, ki])
                    sr_no_grad.append(sr_k.detach())

        # --- Step 2: each condition is an independent update ---
        for ki in range(k):
            wc_k, sr_k = model(lq[:, ki], conditions[:, ki])  # (B,4,H,W), (B,1,2H,2W)

            # L1 reconstruction loss
            l1_k = F.l1_loss(sr_k, gt)
            loss_k = l1_weight * l1_k

            # Wavelet-domain loss
            w_loss_k = torch.tensor(0.0, device=device)
            if wavelet_weight > 0:
                with torch.no_grad():
                    gt_wc = wavelet_transform(gt, wavelet=wavelet_type)  # (B, 4, H, W)
                # Per-subband MSE
                mse = F.mse_loss
                ll_l = mse(wc_k[:, 0:1], gt_wc[:, 0:1])
                lh_l = mse(wc_k[:, 1:2], gt_wc[:, 1:2])
                hl_l = mse(wc_k[:, 2:3], gt_wc[:, 2:3])
                hh_l = mse(wc_k[:, 3:4], gt_wc[:, 3:4])
                w_loss_k = lambda_ll * ll_l + lh_l + hl_l + hh_l
                loss_k = loss_k + wavelet_weight * w_loss_k

            # Consistency loss (against other no_grad views)
            cons_k = torch.tensor(0.0, device=device)
            if cons_weight > 0 and k > 1:
                for kj in range(k):
                    if ki != kj:
                        cons_k = cons_k + F.l1_loss(sr_k, sr_no_grad[kj])
                cons_k = cons_k / (k - 1)
                loss_k = loss_k + cons_weight * cons_k

            optimizer.zero_grad()
            loss_k.backward()
            optimizer.step()

            total_loss += loss_k.item()
            total_l1 += l1_k.item()
            total_wavelet += w_loss_k.item()
            total_cons += cons_k.item()
            n_steps += 1

        pbar.set_postfix({
            'loss': f'{total_loss / n_steps:.4f}',
            'l1': f'{total_l1 / n_steps:.4f}',
            'wav': f'{total_wavelet / n_steps:.4f}',
            'cons': f'{total_cons / n_steps:.4f}',
        })

    return {
        'loss': total_loss / n_steps,
        'l1': total_l1 / n_steps,
        'wavelet': total_wavelet / n_steps,
        'cons': total_cons / n_steps,
    }


@torch.no_grad()
def validate(model, dataloader, config, device):
    """Validate and compute average PSNR & SSIM."""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    n_images = 0
    k = config['degradation']['num_conditions']
    border = config.get('metrics', {}).get('crop_border', 0)

    for batch in tqdm(dataloader, desc='Validating'):
        lq = batch['lq'].to(device)
        gt = batch['gt'].to(device)
        conditions = batch['conditions'].to(device)

        B = lq.size(0)

        for b in range(B):
            for ki in range(k):
                _, sr = model(lq[b:b+1, ki], conditions[b:b+1, ki])
                total_psnr += compute_psnr(sr[0], gt[b], border=border)
                total_ssim += compute_ssim(sr[0], gt[b], border=border)
                n_images += 1

    avg_psnr = total_psnr / n_images if n_images > 0 else 0.0
    avg_ssim = total_ssim / n_images if n_images > 0 else 0.0
    return avg_psnr, avg_ssim


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda / cpu).')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU ids, e.g. "0,1,2,3". '
                             'Default: all visible CUDA devices.')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")

    # Device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Multi-GPU: use the requested (or all visible) GPUs; batch_size is
    # multiplied by the number of GPUs (one image per GPU per step)
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
    elif device == 'cuda' and torch.cuda.device_count() > 1:
        gpu_ids = list(range(torch.cuda.device_count()))
    effective_bs = config['train']['batch_size']
    if gpu_ids:
        device = f'cuda:{gpu_ids[0]}'
        if len(gpu_ids) > 1:
            effective_bs = config['train']['batch_size'] * len(gpu_ids)
            print(f"Multi-GPU: {gpu_ids}, effective batch_size={effective_bs}")
        else:
            print(f"Single GPU: {gpu_ids[0]}")

    # Create save dirs
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    # Acquire the per-directory lock before any checkpoint is written
    train_lock = TrainingLock(
        config['logging']['save_dir'], os.path.basename(config_path))
    train_lock.acquire()
    atexit.register(train_lock.release)

    # Text logger
    logger = TextLogger(config['logging']['log_dir'])
    logger.log(f"Config: {config_path}")
    logger.log(f"Batch size: {effective_bs} ({config['train']['batch_size']} per GPU), "
               f"k: {config['degradation']['num_conditions']}")

    # Datasets
    train_ds = MiceBDataset(config, split='train')
    val_ds = MiceBDataset(config, split='val')
    logger.log(f"Train images: {len(train_ds)}, Val images: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=effective_bs,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # validate one image at a time
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
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

    pc = model.count_parameters()
    active_str = f"Model #Params: {pc['active'] / 1e6:.2f} M"
    if pc['total'] != pc['active']:
        disabled = []
        if not model.use_wavelet:
            disabled.append(f"wavelet={pc['wavelet'] / 1e6:.2f}M")
        if not model.use_cata:
            disabled.append(f"CATA={pc['cata'] / 1e6:.2f}M")
        active_str += f" (excl. {', '.join(disabled)}, shared={pc['shared'] / 1e6:.2f}M)"
    logger.log(active_str)
    logger.log(f"Branches: wavelet={'ON' if model.use_wavelet else 'OFF'}, "
               f"spatial/CATA={'ON' if model.use_cata else 'OFF'}")
    print(active_str)
    print(f"Branches: wavelet={'ON' if model.use_wavelet else 'OFF'}, "
          f"spatial/CATA={'ON' if model.use_cata else 'OFF'}")

    # Multi-GPU wrapper
    if gpu_ids and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        print(f"Wrapped model with nn.DataParallel on {gpu_ids}")

    # Optimizer
    optim_cfg = config['train']['optim']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optim_cfg['lr'],
        weight_decay=optim_cfg.get('weight_decay', 0),
        betas=optim_cfg.get('betas', [0.9, 0.99]),
    )

    # Scheduler
    sched_cfg = config['train']['scheduler']
    if sched_cfg['type'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sched_cfg['T_max'], eta_min=sched_cfg['eta_min']
        )
    elif sched_cfg['type'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=sched_cfg['milestones'], gamma=sched_cfg['gamma']
        )
    else:
        scheduler = None
        print(f"Unknown scheduler type: {sched_cfg['type']}, skipping.")

    # Resume
    start_epoch = 1
    best_psnr = 0.0
    no_improve = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('best_psnr', 0.0)
        no_improve = ckpt.get('no_improve', 0)

    # Training loop
    num_epochs = config['train']['num_epochs']
    val_freq = config['train']['val_freq']
    save_freq = config['logging']['save_freq']
    early_stop = config['train'].get('early_stop', False)
    patience = config['train'].get('early_stop_patience', 20)

    def model_state():
        """Return the raw model state dict (no 'module.' prefix)."""
        return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    for epoch in range(start_epoch, num_epochs + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, config, device, epoch)

        if scheduler:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Log to file
        logger.log_metrics(epoch, {
            'loss': train_metrics['loss'],
            'l1': train_metrics['l1'],
            'wavelet': train_metrics['wavelet'],
            'cons': train_metrics['cons'],
            'lr': current_lr,
        })

        print(f"[Epoch {epoch}/{num_epochs}] "
              f"loss={train_metrics['loss']:.4f} "
              f"l1={train_metrics['l1']:.4f} "
              f"wav={train_metrics['wavelet']:.4f} "
              f"cons={train_metrics['cons']:.4f} "
              f"lr={current_lr:.2e}")

        # Validate
        if epoch % val_freq == 0:
            val_psnr, val_ssim = validate(model, val_loader, config, device)
            logger.log_metrics(epoch, {'val_psnr': val_psnr, 'val_ssim': val_ssim})
            print(f"  -> Val PSNR: {val_psnr:.2f} dB  |  SSIM: {val_ssim:.4f}")

            # Save best
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                no_improve = 0
                best_path = os.path.join(config['logging']['save_dir'], 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'model': model_state(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'best_psnr': best_psnr,
                    'no_improve': no_improve,
                    'config': config,
                }, best_path)
                print(f"  -> Saved best model (PSNR={best_psnr:.2f} dB) to {best_path}")
            else:
                no_improve += val_freq
                print(f"  -> No improvement ({no_improve}/{patience} epochs)")

            # Early stopping
            if early_stop and no_improve >= patience:
                logger.log(f"Early stopping at epoch {epoch}: no val PSNR "
                           f"improvement for {no_improve} epochs (patience={patience})")
                print(f"  -> Early stopping (patience={patience})")
                break

        # Checkpoint
        if epoch % save_freq == 0:
            ckpt_path = os.path.join(config['logging']['save_dir'], f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model': model_state(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_psnr': best_psnr,
                'no_improve': no_improve,
                'config': config,
            }, ckpt_path)
            print(f"  -> Saved checkpoint to {ckpt_path}")

    # Final save
    final_path = os.path.join(config['logging']['save_dir'], 'final.pth')
    torch.save({
        'epoch': epoch,
        'model': model_state(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'best_psnr': best_psnr,
        'no_improve': no_improve,
        'config': config,
    }, final_path)
    print(f"Training finished. Final model saved to {final_path}")

    logger.log(f"Training finished. Final model: {final_path}")
    logger.close()


if __name__ == '__main__':
    main()
