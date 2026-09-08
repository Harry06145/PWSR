"""Minimal utility functions extracted from BasicSR for PWSR.

Only includes ``trunc_normal_`` — the sole function needed by
the CATANet TAB/LRSA weight initialisation and the PWSR model.

BasicSR (https://github.com/XPixelGroup/BasicSR) is released under the
Apache-2.0 license; see ``licenses/Apache-2.0.txt``.
"""

import math
import warnings
import torch


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds.

    Args:
        tensor: an n-dimensional ``torch.Tensor``
        mean:   the mean of the normal distribution
        std:    the standard deviation of the normal distribution
        a:      the minimum cutoff value
        b:      the maximum cutoff value
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
