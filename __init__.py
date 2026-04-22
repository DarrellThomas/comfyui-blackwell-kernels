# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""ComfyUI custom node — Blackwell Kernels for RTX 5090.

Automatically replaces ComfyUI's attention with a custom Flash Attention
kernel optimized for sm_120a (RTX 5090). No configuration needed —
detects the GPU at startup and activates if supported.
"""

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Only activate on RTX 5090 (sm_120a / compute capability 12.0)
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
            """Drop-in replacement for ComfyUI's optimized_attention.

            Handles the reshape logic ComfyUI expects, then calls our kernel.
            Falls back to the original attention for unsupported configs.
            """
            # ComfyUI passes q/k/v as [B*H, N, D] when skip_reshape=False
            # or [B, H, N, D] when skip_reshape=True
            if not skip_reshape:
                B_times_H, N, D = q.shape
                H = heads
                B = B_times_H // H
                q = q.reshape(B, H, N, D)
                k = k.reshape(B, H, N, D)
                v = v.reshape(B, H, N, D)
            else:
                B, H, N, D = q.shape

            # Our kernel supports D=40, 64, 128 in BF16
            if D not in (40, 64, 128) or q.dtype != torch.bfloat16:
                # Fall back to original for unsupported head dims or dtypes
                if _original_attention is not None:
                    if not skip_reshape:
                        q = q.reshape(B * H, N, D)
                        k = k.reshape(B * H, N, D)
                        v = v.reshape(B * H, N, D)
                    return _original_attention(q, k, v, heads, mask=mask,
                                               attn_precision=attn_precision,
                                               skip_reshape=skip_reshape,
                                               skip_output_reshape=skip_output_reshape,
                                               **kwargs)
                # Last resort: PyTorch SDPA
                import torch.nn.functional as F
                out = F.scaled_dot_product_attention(q, k, v)
                if not skip_output_reshape:
                    out = out.transpose(1, 2).reshape(B, N, H * D)
                return out

            # Causal = False for image generation (no autoregressive masking)
            causal = False

            out = flash_attention(q, k, v, causal=causal)  # [B, H, N, D]

            if not skip_output_reshape:
                # ComfyUI expects [B, N, H*D] output
                out = out.transpose(1, 2).reshape(B, N, H * D)

            return out

        # Monkey-patch ComfyUI's attention dispatcher
        try:
            import comfy.ldm.modules.attention as comfy_attn
            if hasattr(comfy_attn, 'optimized_attention'):
                _original_attention = comfy_attn.optimized_attention
                comfy_attn.optimized_attention = _blackwell_attention
                _ACTIVE = True
                print(f"[Blackwell Kernels] ACTIVE — RTX 5090 Flash Attention replacing SDPA (D=40/64/128)")
            else:
                print(f"[Blackwell Kernels] WARNING — comfy.ldm.modules.attention.optimized_attention not found")
                print(f"[Blackwell Kernels] Kernel loaded but not patched into ComfyUI")
        except ImportError:
            print(f"[Blackwell Kernels] WARNING — ComfyUI not detected (standalone mode)")
            print(f"[Blackwell Kernels] Kernel available via: from blackwell_kernels import flash_attention")

    except ImportError as e:
        print(f"[Blackwell Kernels] Extension not compiled: {e}")
        print(f"[Blackwell Kernels] Build with: CUDA_HOME=/usr/local/cuda-13 pip install -e .")
