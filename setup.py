# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

import os
from setuptools import setup, find_packages

# Bypass PyTorch's CUDA version check — we need CUDA 13.2 nvcc for sm_120a
# but PyTorch wheels ship with CUDA 12.6 runtime (which is forward-compatible)
os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")

# Monkey-patch the CUDA version check before importing CUDAExtension
import torch.utils.cpp_extension as _cpp_ext
_orig_check = getattr(_cpp_ext, '_check_cuda_version', None)
if _orig_check:
    _cpp_ext._check_cuda_version = lambda *a, **kw: None

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="comfyui-blackwell-kernels",
    version="0.1.0",
    description="RTX 5090 (sm_120a) CUDA kernels for ComfyUI — Flash Attention, GroupNorm, MLA",
    author="Darrell Thomas",
    license="MIT",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            "blackwell_kernels._C",
            [
                "csrc/attention/flash_attn_sm120a.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode", "arch=compute_120a,code=sm_120a",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",
                ],
            },
            include_dirs=[
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "common"),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch>=2.5"],
)
