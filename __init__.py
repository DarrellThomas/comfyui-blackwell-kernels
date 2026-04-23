# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""ComfyUI custom node — Blackwell Kernels for RTX 5090."""

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_SUPPORTED = False
_ACTIVE = False
_original_attention = None

if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    if cc >= (12, 0):
        _SUPPORTED = True

if not _SUPPORTED:
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        print(f"[Blackwell Kernels] Skipped — GPU is CC {cc[0]}.{cc[1]}, need >= 12.0 (RTX 5090)")
    else:
        print("[Blackwell Kernels] Skipped — no CUDA GPU detected")
else:
    try:
        from blackwell_kernels.attention import flash_attention

        def _blackwell_attention(q, k, v, heads, mask=None, attn_precision=None,
                                 skip_reshape=False, skip_output_reshape=False, **kwargs):
            """Drop-in for ComfyUI's optimized_attention.

            Mirrors the exact reshape logic from ComfyUI's attention_pytorch:
              skip_reshape=False: q is [B, N, D_total], reshape to [B, H, N, d_head]
              skip_reshape=True:  q is already [B, H, N, d_head]
            """
            if skip_reshape:
                if q.ndim == 4:
                    b, _, _, dim_head = q.shape
                else:
                    # Some paths pass 3D even with skip_reshape=True
                    b, _, dim_head = q.shape
                    dim_head //= heads
                    q = q.reshape(b, -1, heads, dim_head).transpose(1, 2)
                    k = k.reshape(b, -1, heads, dim_head).transpose(1, 2)
                    v = v.reshape(b, -1, heads, dim_head).transpose(1, 2)
            else:
                b, _, dim_head = q.shape
                dim_head //= heads
                q = q.reshape(b, -1, heads, dim_head).transpose(1, 2)
                k = k.reshape(b, -1, heads, dim_head).transpose(1, 2)
                v = v.reshape(b, -1, heads, dim_head).transpose(1, 2)

            # Now q/k/v are [B, H, N, dim_head]
            B, H, N, D = q.shape

            # Only use our kernel for supported dims + BF16
            if D in (40, 64, 128) and q.dtype == torch.bfloat16:
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
                out = flash_attention(q, k, v, causal=False)
            else:
                # Fall back to PyTorch SDPA
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask)

            # [B, H, N, D] -> output format
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * D)
            return out

        # Monkey-patch
        try:
            import comfy.ldm.modules.attention as comfy_attn
            if hasattr(comfy_attn, 'optimized_attention'):
                _original_attention = comfy_attn.optimized_attention
                comfy_attn.optimized_attention = _blackwell_attention
                _ACTIVE = True
                print(f"[Blackwell Kernels] ACTIVE — RTX 5090 Flash Attention replacing SDPA (D=40/64/128)")
            else:
                print(f"[Blackwell Kernels] WARNING — optimized_attention not found")
        except ImportError:
            print(f"[Blackwell Kernels] WARNING — ComfyUI not detected (standalone mode)")

    except ImportError as e:
        print(f"[Blackwell Kernels] Extension not compiled: {e}")
        print(f"[Blackwell Kernels] Build with: CUDA_HOME=/usr/local/cuda-13 pip install -e .")
