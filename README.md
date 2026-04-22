# ComfyUI Blackwell Kernels

Custom CUDA kernels for **RTX 5090** that make ComfyUI image generation faster. Drop-in replacement — no workflow changes needed.

## What it does

Replaces ComfyUI's default attention (cuDNN SDPA) with a hand-tuned Flash Attention kernel built specifically for the RTX 5090's tensor cores. The kernel activates automatically when it detects your GPU.

**Performance vs stock ComfyUI:**

| Head Dim | Causal | Speedup vs cuDNN |
|----------|--------|------------------|
| D=40     | No     | **1.22x faster** |
| D=64     | Yes    | **1.07x faster** |
| D=64     | No     | ~1.00x (parity)  |
| D=128    | Yes    | **1.05x faster** |

These are kernel-level speedups. Real-world image generation improvement depends on how much time is spent in attention (typically 30-60% of total).

## Requirements

- **GPU:** NVIDIA RTX 5090 (Compute Capability 12.0)
- **OS:** Windows 10/11 or Linux
- **CUDA Toolkit:** 13.2 ([download](https://developer.nvidia.com/cuda-13-2-0-download-archive))
- **Python:** 3.10+
- **PyTorch:** 2.5+ with CUDA support
- **ComfyUI:** Any recent version

## Installation

### Option 1: Pre-built wheel (easiest)

Download the wheel for your OS from [Releases](https://github.com/DarrellThomas/comfyui-blackwell-kernels/releases):

```bash
pip install comfyui_blackwell_kernels-0.1.0-*.whl
```

Then clone or download this repo into ComfyUI's custom nodes:

```
ComfyUI/
  custom_nodes/
    comfyui-blackwell-kernels/    <-- this repo
      __init__.py
      python/
      ...
```

### Option 2: Build from source (Windows)

1. **Install CUDA 13.2 Toolkit** from [NVIDIA](https://developer.nvidia.com/cuda-13-2-0-download-archive)
   - Choose "Custom" install, select only "CUDA Toolkit" (you don't need the driver update)

2. **Install Visual Studio Build Tools** (if you don't have them):
   - Download from [Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Select "Desktop development with C++"

3. **Clone and build:**
   ```cmd
   cd ComfyUI\custom_nodes
   git clone https://github.com/DarrellThomas/comfyui-blackwell-kernels.git
   cd comfyui-blackwell-kernels
   build.bat
   ```

### Option 3: Build from source (Linux)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DarrellThomas/comfyui-blackwell-kernels.git
cd comfyui-blackwell-kernels
CUDA_HOME=/usr/local/cuda-13 ./build.sh
```

## Verifying it works

Start ComfyUI normally. In the terminal/console output, you should see:

```
[Blackwell Kernels] ACTIVE — RTX 5090 Flash Attention replacing SDPA (D=40/64/128)
```

If you see this, the kernels are active and every attention operation in your workflows will use them automatically. No workflow changes needed.

**If you see errors instead:**

| Message | Fix |
|---------|-----|
| `Extension not compiled` | Run `build.bat` or `build.sh` |
| `Skipped — GPU is CC X.Y` | You need an RTX 5090 (CC 12.0) |
| `optimized_attention not found` | Your ComfyUI version may be too old — update ComfyUI |

## A/B Performance Test

Run this to see the speed difference on your machine:

```bash
cd ComfyUI/custom_nodes/comfyui-blackwell-kernels
python benchmark.py
```

This runs identical attention operations with stock PyTorch (cuDNN SDPA) and our custom kernel, side by side, and prints a comparison table.

## How it works (technical)

The custom node monkey-patches `comfy.ldm.modules.attention.optimized_attention` at startup. Every attention call in every workflow — text encoders, U-Net, VAE — goes through our kernel instead of cuDNN.

**Supported head dimensions:** 40, 64, 128 (covers SD 1.5, SDXL, Flux, and most models). Unsupported dimensions fall back to stock PyTorch automatically.

**Data format:** BF16 only (ComfyUI's default for RTX cards). FP16/FP32 inputs fall back to stock.

## Disabling

To temporarily disable without uninstalling, rename the folder:

```
mv comfyui-blackwell-kernels comfyui-blackwell-kernels.disabled
```

ComfyUI will skip it on next restart.

## License

MIT — Darrell Thomas, 2026
