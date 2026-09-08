"""
PWSR: Physical-aware Parallel Dual-Domain Network
for Robust NIR-II Fluorescence Super-Resolution.

Encodes (λ, NA) with an MLP, spatially tiles the embedding,
and concatenates it with the input image as extra channels.

After condition concatenation, two parallel branches:
  - Wavelet branch: heavy CNN at H/2 × W/2 (wavelet domain, cheap)
  - Spatial branch: CATANet TAB blocks at H × W (spatial domain)

Outputs are fused via a learnable convolution head (concat → conv).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from catanet import TAB, LRSA
from catanet.arch_util import trunc_normal_


# ============================================================================
# Wavelet Transform Utilities
# ============================================================================

def wt(data, wavelet='haar'):
    """Forward 2D discrete wavelet transform (single level).

    Returns (B, 4C, H/2, W/2): [LL, LH, HL, HH].
    """
    import ptwt
    LL, (HL, LH, HH) = ptwt.wavedec2(data, wavelet=wavelet, level=1)
    return torch.cat([LL, HL, LH, HH], dim=1)


def iwt(data, wavelet='haar'):
    """Inverse 2D discrete wavelet transform (single level).

    Input (B, 4C, H/2, W/2) -> Output (B, C, H, W).
    """
    import ptwt
    channels = data.shape[1] // 4
    LL = data[:, :channels, :, :]
    HL = data[:, channels:channels * 2, :, :]
    LH = data[:, channels * 2:channels * 3, :, :]
    HH = data[:, channels * 3:, :, :]
    return ptwt.waverec2((LL, (HL, LH, HH)), wavelet=wavelet)


# ============================================================================
# Condition MLP
# ============================================================================

class ConditionMLP(nn.Module):
    """Maps normalized physical parameters (λ, NA) ∈ [0,1]²
    to a condition embedding vector.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: list = None,
                 cond_dim: int = 16, act=nn.GELU):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64]
        layers = []
        in_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            layers.append(act())
            in_dim = hd
        layers.append(nn.Linear(in_dim, cond_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        return self.mlp(conditions)


# ============================================================================
# Wavelet Residual Block (GELU — preserves negative coefficients)
# ============================================================================

class WaveletResBlock(nn.Module):
    """Residual block with GELU for wavelet domain.

    Wavelet subbands (LH, HL, HH) contain both positive and negative
    coefficients, so ReLU is unsuitable. GELU allows negative values.
    """

    def __init__(self, num_feat: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        return x + self.conv2(self.gelu(self.conv1(x)))


# ============================================================================
# PWSR: Physical-aware Parallel Dual-Domain Super-Resolution Network
# ============================================================================

class PWSR(nn.Module):
    """PWSR with dual-domain (wavelet + spatial) parallel learning.

    Architecture::

        Input: lq (B, 1, H, W) + conditions (B, 2)
          |
          +--> Condition MLP → spatial tile → concat → x (B, 1+cond_dim, H, W)
          |
          +--> Wavelet Branch (H/2 × W/2, heavy CNN):
          |      wt(x) → conv → ResBlocks × N → PixelShuffle(2) → wavelet_coeffs
          |        |
          |        v
          |      iwt → sr_wavelet (B, 1, 2H, 2W)
          |
          +--> Spatial Branch (H × W, CATANet):
          |      conv → TAB+LRSA blocks × N → PixelShuffle(2) → sr_spatial
          |
          v
        Fusion: learned conv fusion of sr_wavelet and sr_spatial (B, 1, 2H, 2W)

    Returns:
        (wavelet_coeffs, sr) where wavelet_coeffs has shape (B, 4, H, W).
    """

    def __init__(self,
                 cond_dim: int = 16,
                 cond_input_dim: int = 2,
                 hidden_dims: list = None,
                 use_checkpoint: bool = True,
                 # Image I/O
                 in_chans: int = 1,
                 upscale: int = 2,
                 # Shared feature dim
                 dim: int = 40,
                 # Wavelet branch
                 wavelet: str = 'haar',
                 wavelet_blocks: int = 8,
                 # Spatial (CATANet) branch
                 block_num: int = 8,
                 qk_dim: int = 36,
                 mlp_dim: int = 96,
                 heads: int = 4,
                 patch_size: list = None,
                 n_iters: list = None,
                 num_tokens: list = None,
                 group_size: list = None,
                 # Ablation switches
                 use_wavelet: bool = True,
                 use_cata: bool = True,
                 ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64]

        self.cond_dim = cond_dim
        self.in_chans = in_chans
        self.upscale = upscale
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.wavelet_type = wavelet
        self.wavelet_blocks = wavelet_blocks
        self.block_num = block_num
        self.use_wavelet = use_wavelet
        self.use_cata = use_cata

        # Total input channels after condition concatenation
        total_in_chans = in_chans + cond_dim

        # --- Condition MLP ---
        self.cond_mlp = ConditionMLP(
            input_dim=cond_input_dim, hidden_dims=hidden_dims, cond_dim=cond_dim
        )

        # ============ Wavelet Branch (H/2 × W/2, heavy) ============
        # Wavelet transform produces 4× channels at half resolution
        wt_ch = 4 * total_in_chans

        self.wavelet_input = nn.Sequential(
            nn.Conv2d(wt_ch, dim, 3, 1, 1),
            nn.GELU()
        )

        # Wavelet body: GELU residual blocks at H/2 × W/2
        w_body = []
        for _ in range(wavelet_blocks):
            w_body.append(WaveletResBlock(num_feat=dim))
        self.wavelet_body = nn.Sequential(*w_body)

        # Wavelet upsampling: dim → dim*(upscale²) → PixelShuffle → dim → 4·in_chans
        # PixelShuffle(upscale) in wavelet domain: H/2 → H·upscale/2
        # Then iwt doubles resolution: H·upscale/2 → H·upscale
        assert upscale == 2, "Currently only upscale=2 is supported"
        self.wavelet_up = nn.Sequential(
            nn.Conv2d(dim, dim * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.GELU(),
            nn.Conv2d(dim, 4 * in_chans, 3, 1, 1)
        )

        # ============ Spatial Branch (H × W, CATANet TAB blocks) ============
        self.spatial_input = nn.Conv2d(total_in_chans, dim, 3, 1, 1)

        # Default per-block settings
        if n_iters is None:
            n_iters = [5] * block_num
        if num_tokens is None:
            num_tokens = [16, 32, 64, 128, 16, 32, 64, 128][:block_num]
        if group_size is None:
            group_size = [256, 128, 64, 32, 256, 128, 64, 32][:block_num]
        if patch_size is None:
            patch_size = [16, 20, 24, 28, 16, 20, 24, 28][:block_num]

        self.patch_size = patch_size

        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()

        for i in range(block_num):
            self.blocks.append(nn.ModuleList([
                TAB(dim, qk_dim, mlp_dim, heads, n_iters[i],
                    num_tokens[i], group_size[i], dropout=0),  # attention dropout disabled
                LRSA(dim, qk_dim, mlp_dim, heads)
            ]))
            self.mid_convs.append(nn.Conv2d(dim, dim, 3, 1, 1))

        # Spatial upsampling: dim → dim*(upscale²) → PixelShuffle → in_chans
        if upscale == 2:
            self.spatial_up = nn.Sequential(
                nn.Conv2d(dim, dim * (upscale ** 2), 3, 1, 1),
                nn.PixelShuffle(upscale),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(dim, in_chans, 3, 1, 1)
            )
        else:
            self.spatial_up = nn.Sequential(
                nn.Conv2d(dim, dim * (upscale ** 2), 3, 1, 1),
                nn.PixelShuffle(upscale),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(dim, in_chans, 3, 1, 1)
            )

        # Learned fusion head: concatenate both branch outputs and fuse
        # with a small convolution (instead of simple addition).
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * in_chans, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, in_chans, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _forward_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """CATANet TAB+LRSA blocks at LQ resolution (H × W).

        Each block: TAB (global attention) → LRSA (local attention)
        with per-block residual connection.
        """
        for i in range(self.block_num):
            residual = x
            global_attn, local_attn = self.blocks[i]
            x = global_attn(x)
            x = local_attn(x, self.patch_size[i])
            x = residual + self.mid_convs[i](x)
        return x

    def count_parameters(self) -> dict:
        """Count parameters per branch.

        Returns dict with keys:
            total   — all parameters in the model
            shared  — condition MLP (always active)
            wavelet — wavelet branch only
            cata    — spatial/CATA branch only
            active  — parameters in currently active branches
        """
        def _count(*modules):
            return sum(p.numel() for m in modules for p in m.parameters())

        w_params = _count(self.wavelet_input, self.wavelet_body, self.wavelet_up)
        c_params = _count(self.spatial_input, self.blocks, self.mid_convs, self.spatial_up)
        s_params = _count(self.cond_mlp)
        f_params = _count(self.fusion)

        active = s_params
        if self.use_wavelet:
            active += w_params
        if self.use_cata:
            active += c_params
        if self.use_wavelet and self.use_cata:
            active += f_params

        return {
            'total': s_params + w_params + c_params + f_params,
            'shared': s_params,
            'wavelet': w_params,
            'cata': c_params,
            'fusion': f_params,
            'active': active,
        }

    def forward(self, lq: torch.Tensor, conditions: torch.Tensor):
        """
        Args:
            lq:         (B, 1, H, W) degraded image.
            conditions: (B, 2) normalized (lambda, NA).

        Returns:
            wavelet_coeffs: (B, 4, H, W) — predicted wavelet subbands
                            at LQ resolution, for wavelet-domain loss.
            sr:             (B, 1, H * upscale, W * upscale) — super-resolved image.
        """
        B, _, H, W = lq.shape

        # ---- Encode & concatenate conditions ----
        cond_embed = self.cond_mlp(conditions)                      # (B, cond_dim)
        cond_map = cond_embed[:, :, None, None].expand(B, self.cond_dim, H, W)
        x = torch.cat([lq, cond_map], dim=1)                        # (B, total_in_chans, H, W)

        # ==== Wavelet Branch (H/2 × W/2, heavy CNN) ====
        if self.use_wavelet:
            x_wt = wt(x, wavelet=self.wavelet_type)                 # (B, 4 * total_in_chans, H/2, W/2)
            x_wt = self.wavelet_input(x_wt)                          # (B, dim, H/2, W/2)
            x_wt = self.wavelet_body(x_wt)                           # (B, dim, H/2, W/2)
            wavelet_coeffs = self.wavelet_up(x_wt)                   # (B, 4 * in_chans, H, W)
            sr_wavelet = iwt(wavelet_coeffs, wavelet=self.wavelet_type)  # (B, in_chans, 2H, 2W)
        else:
            wavelet_coeffs = torch.zeros(B, 4 * self.in_chans, H, W,
                                         device=lq.device, dtype=lq.dtype)
            sr_wavelet = None

        # ==== Spatial Branch (H × W, CATANet TAB blocks) ====
        if self.use_cata:
            feat = self.spatial_input(x)                             # (B, dim, H, W)

            if self.use_checkpoint and self.training:
                feat = torch_checkpoint(self._forward_spatial_features, feat, use_reentrant=False)
            else:
                feat = self._forward_spatial_features(feat)

            feat = feat + self.spatial_input(x)                      # outer residual
            sr_spatial = self.spatial_up(feat)                       # (B, in_chans, 2H, 2W)
        else:
            sr_spatial = None

        # ==== Fusion: learned conv fusion when both branches are active;
        #      pass through the single active branch otherwise. ====
        if self.use_wavelet and self.use_cata:
            sr = self.fusion(torch.cat([sr_wavelet, sr_spatial], dim=1))
        elif self.use_wavelet:
            sr = sr_wavelet
        elif self.use_cata:
            sr = sr_spatial
        else:
            raise ValueError("At least one of use_wavelet or use_cata must be True")

        return wavelet_coeffs, sr


# ============================================================================
# Quick test
# ============================================================================
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = PWSR(
        cond_dim=16, hidden_dims=[32, 64],
        in_chans=1, upscale=2, use_checkpoint=False,
        dim=40, wavelet_blocks=8, block_num=8,
    ).to(device)

    lq = torch.randn(2, 1, 256, 256).to(device)
    cond = torch.rand(2, 2).to(device)

    with torch.no_grad():
        wc, out = model(lq, cond)

    print(f"Input:            {lq.shape}")
    print(f"Conditions:       {cond.shape}")
    print(f"Wavelet coeffs:   {wc.shape}")
    print(f"SR output:        {out.shape}")
    print(f"#Params:          {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    print(f"\n=== Architecture ===")
    print(f"Wavelet branch: {model.wavelet_blocks} ResBlocks @ H/2×W/2 (128×128)")
    print(f"Spatial branch: {model.block_num} TAB+LRSA blocks @ H×W (256×256)")
    print(f"Fusion: learned conv fusion of the two branch outputs")
