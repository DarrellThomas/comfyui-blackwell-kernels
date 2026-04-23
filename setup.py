# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

import os
import sys
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

# Platform-specific C++ compiler flags
if sys.platform == "win32":
    # /D_HAS_STD_BYTE=0 fixes 'std' ambiguous symbol in PyTorch headers on MSVC
    cxx_flags = ["/O2", "/std:c++17", "/D_HAS_STD_BYTE=0"]
    # Windows uses CUDA 12.8 (sm_120, no CCCL conflicts with PyTorch)
    cuda_arch = "compute_120"
    sm_code = "sm_120"
else:
    cxx_flags = ["-O3", "-std=c++17"]
    # Linux uses CUDA 13.2 (sm_120a)
    cuda_arch = "compute_120a"
    sm_code = "sm_120a"

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
                "cxx": cxx_flags,
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode", f"arch={cuda_arch},code={sm_code}",
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
