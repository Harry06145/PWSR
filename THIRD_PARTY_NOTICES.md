# Third-Party Notices

This repository includes code derived from third-party open-source projects.
The corresponding license texts are kept under `licenses/`.

## CATANet

The spatial-domain attention blocks (`catanet/catanet_arch.py`) are adapted
from the official implementation of
[CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image
Super-Resolution](https://github.com/EquationWalker/CATANet) (Liu et al.,
CVPR 2025), licensed under the Apache License 2.0.

## BasicSR

`catanet/arch_util.py` contains a minimal `trunc_normal_` utility extracted
from [BasicSR](https://github.com/XPixelGroup/BasicSR), licensed under the
Apache License 2.0.

## PyTorch-Wavelet-Toolbox (dependency)

The wavelet transform is provided by
[ptwt](https://github.com/v0lta/PyTorch-Wavelet-Toolbox) (MIT license), which
is installed as a dependency rather than vendored.

## License Texts

- `licenses/Apache-2.0.txt` — Apache License 2.0 (CATANet, BasicSR).

The remainder of the code is released under the MIT License; see `LICENSE`.
