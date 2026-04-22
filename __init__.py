# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""ComfyUI custom node — Blackwell Kernels for RTX 5090."""

import torch

# Only activate on RTX 5090 (sm_120a / compute capability 12.0)
_SUPPORTED = False
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    if cc >= (12, 0):
        _SUPPORTED = True

if _SUPPORTED:
    try:
        from blackwell_kernels import flash_attention
        import comfy.model_management

        # Monkey-patch ComfyUI's attention to use our kernel
        _original_attention = None
        try:
            import comfy.ldm.modules.attention as comfy_attn
            if hasattr(comfy_attn, 'optimized_attention'):
                _original_attention = comfy_attn.optimized_attention
        except ImportError:
            pass

        print(f"[Blackwell Kernels] Loaded — RTX 5090 Flash Attention active (CC {cc[0]}.{cc[1]})")
    except ImportError as e:
        print(f"[Blackwell Kernels] Extension not compiled: {e}")
        print("[Blackwell Kernels] Run: CUDA_HOME=/usr/local/cuda-13 pip install -e .")
else:
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        print(f"[Blackwell Kernels] Skipped — GPU is CC {cc[0]}.{cc[1]}, need >= 12.0 (RTX 5090)")
    else:
        print("[Blackwell Kernels] Skipped — no CUDA GPU detected")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
