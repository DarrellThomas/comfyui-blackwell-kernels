# Installing Blackwell Kernels on Windows — Step by Step

This guide walks you through everything from scratch. If you already have some of these installed, skip those steps.

---

## Step 1: Install Python 3.12

1. Go to https://www.python.org/downloads/
2. Click the big yellow **"Download Python 3.12.x"** button
3. Run the installer
4. **IMPORTANT:** Check the box that says **"Add Python to PATH"** at the bottom of the first screen
5. Click **"Install Now"**
6. When it finishes, click **"Close"**

**Verify it worked:**
- Press `Win + R`, type `cmd`, press Enter
- Type `python --version` and press Enter
- You should see `Python 3.12.x`

---

## Step 2: Install Git

1. Go to https://git-scm.com/downloads/win
2. Click **"Click here to download"** (the 64-bit version)
3. Run the installer
4. Click **Next** through everything (the defaults are fine)
5. Click **Install**, then **Finish**

**Verify it worked:**
- Open a new Command Prompt (`Win + R`, type `cmd`, press Enter)
- Type `git --version` and press Enter
- You should see `git version 2.x.x`

---

## Step 3: Install CUDA 13.2 Toolkit

This is the NVIDIA compiler toolkit. You already have the GPU driver — this is the development tools.

1. Go to https://developer.nvidia.com/cuda-13-2-0-download-archive
2. Select: **Windows** → **x86_64** → **11** (or your Windows version) → **exe (local)**
3. Download the installer (~3 GB, this takes a few minutes)
4. Run the installer
5. When it asks, choose **Custom** installation
6. **Uncheck everything EXCEPT "CUDA Toolkit"** (you don't need the driver update, samples, or Visual Studio integration)
7. Click **Next** → **Install**
8. When it finishes, click **Close**

**Verify it worked:**
- Open a **new** Command Prompt
- Type `nvcc --version` and press Enter
- You should see something like `Cuda compilation tools, release 13.2`

**If `nvcc` is not found:** The installer may not have added it to your PATH. You can fix this:
1. Open **Settings** → **System** → **About** → **Advanced system settings** → **Environment Variables**
2. Under **System variables**, find **Path** and click **Edit**
3. Click **New** and add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin`
4. Click **OK** on everything
5. Open a **new** Command Prompt and try `nvcc --version` again

---

## Step 4: Install Visual Studio Build Tools

This gives you the C++ compiler that CUDA needs.

1. Go to https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Click **"Download Build Tools"**
3. Run the installer
4. In the installer window, check **"Desktop development with C++"**
5. Click **Install** (this downloads ~2 GB)
6. When it finishes, you can close the installer

---

## Step 5: Install PyTorch

Open a Command Prompt and run:

```
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

This downloads ~2.5 GB. Wait for it to finish.

**Verify it worked:**
```
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name())"
```

You should see your PyTorch version, `CUDA: True`, and `NVIDIA GeForce RTX 5090`.

---

## Step 6: Download and Build the Kernels

1. Open a Command Prompt
2. Navigate to your ComfyUI custom_nodes folder:
   ```
   cd C:\path\to\ComfyUI\custom_nodes
   ```
   (Replace `C:\path\to\ComfyUI` with wherever your ComfyUI is installed. Common locations:
   - `C:\ComfyUI\custom_nodes`
   - `C:\Users\YourName\ComfyUI\custom_nodes`
   - `D:\ComfyUI\custom_nodes`)

3. Clone the repository:
   ```
   git clone https://github.com/DarrellThomas/comfyui-blackwell-kernels.git
   ```

4. Go into the folder:
   ```
   cd comfyui-blackwell-kernels
   ```

5. Build the kernels:
   ```
   build.bat
   ```

   This compiles the CUDA code for your RTX 5090. It takes **2-5 minutes** — you'll see a lot of compiler output scrolling by. That's normal.

   When it's done, you should see:
   ```
   === BUILD SUCCESSFUL ===
   ```

**If the build fails**, see [Troubleshooting](#troubleshooting) below.

---

## Step 7: Verify It Works

Before starting ComfyUI, run the benchmark test:

```
python benchmark.py
```

This runs a quick speed test comparing stock PyTorch attention vs the custom kernels. You should see a table like:

```
==============================================================================
Config                           Stock (ms)  Blackwell (ms)    Speedup
------------------------------------------------------------------------------
SD 1.5 self-attn (D=40)              0.045           0.037      1.22x  FASTER
SD 1.5 cross-attn (D=64)             0.032           0.030      1.07x  FASTER
SDXL self-attn (D=64)                0.058           0.056      1.04x  FASTER
SDXL cross-attn (D=128)              0.095           0.090      1.05x  FASTER
Flux (D=128, long seq)               0.180           0.171      1.05x  FASTER
==============================================================================
```

If you see "FASTER" in the rightmost column, everything is working.

---

## Step 8: Start ComfyUI

Start ComfyUI the way you normally do. In the terminal/console window, look for this line near the top:

```
[Blackwell Kernels] ACTIVE — RTX 5090 Flash Attention replacing SDPA (D=40/64/128)
```

**That's it!** Every image you generate now uses the faster kernels automatically. No workflow changes needed — your existing workflows, models, and settings all work exactly the same, just faster.

---

## Troubleshooting

### "Python is not recognized"
You didn't check "Add Python to PATH" during installation. The easiest fix: uninstall Python, reinstall, and make sure to check that box.

### "git is not recognized"
Close your Command Prompt and open a new one. If it still doesn't work, restart your computer.

### "nvcc is not recognized"
See the PATH fix in Step 3 above.

### "error: Microsoft Visual C++ 14.0 or greater is required"
You need to install Visual Studio Build Tools (Step 4). Make sure you selected "Desktop development with C++".

### "CUDA_HOME environment variable is not set"
Open a Command Prompt and run:
```
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
build.bat
```

### "RuntimeError: CUDA version mismatch"
This is normal — the build script handles this automatically. If you see it as a fatal error, make sure you're using the `build.bat` from this repository (not running `pip install` directly).

### "Extension not compiled" when starting ComfyUI
The build step didn't complete. Go back to Step 6 and run `build.bat` again. Look for `BUILD SUCCESSFUL` at the end.

### Build takes forever (more than 10 minutes)
CUDA compilation is slow. 5 minutes is normal. If it's been more than 15 minutes, something may be stuck — press `Ctrl+C` to cancel and try again.

### ComfyUI doesn't show the "[Blackwell Kernels] ACTIVE" message
1. Make sure the folder is in the right place: `ComfyUI/custom_nodes/comfyui-blackwell-kernels/`
2. Make sure `build.bat` completed successfully
3. Restart ComfyUI completely (close and reopen)

### "Skipped — GPU is CC X.Y, need >= 12.0"
This means you don't have an RTX 5090. These kernels only work on RTX 5090 (and future Blackwell consumer GPUs).

---

## Using the Kernels

**There's nothing to change in your workflows.** The kernels work automatically behind the scenes.

Once you see `[Blackwell Kernels] ACTIVE` in the ComfyUI console, every image generation uses the faster kernels. Here's what that means:

### What stays the same
- All your existing workflows work exactly as before
- All your models (SD 1.5, SDXL, Flux, etc.) work exactly as before
- All your settings, samplers, and schedulers work exactly as before
- The ComfyUI interface looks exactly the same

### What changes
- **Image generation is faster.** The attention operations (the math-heavy part inside the model) run on custom code optimized for your RTX 5090's hardware instead of the generic NVIDIA library.
- **The console shows it's active.** When ComfyUI starts, you'll see the `[Blackwell Kernels] ACTIVE` message. That's how you know it's working.

### How to see the speed difference

The easiest way to compare:

1. **With kernels enabled:** Generate an image with your favorite workflow. Note the time shown in ComfyUI (bottom of the screen or in the console).
2. **With kernels disabled:** Rename the folder to `comfyui-blackwell-kernels.disabled` (see [Disabling](#disabling) below), restart ComfyUI, generate the same image with the same settings, and note the time.
3. **Compare.** The version with kernels should be faster, especially on models with D=40 heads (like SD 1.5).

### Which models benefit most

| Model | Head Dimension | Expected Speedup |
|-------|---------------|-----------------|
| SD 1.5 | D=40 | **~22% faster** attention |
| SDXL | D=64 | **~7% faster** attention (causal) |
| Flux | D=128 | **~5% faster** attention |

The overall image generation speedup depends on your workflow. Attention is typically 30-60% of total generation time, so a 10% attention speedup translates to roughly 3-6% faster images. More complex workflows with more attention steps see a bigger benefit.

---

## Updating

When a new version is released:

```
cd C:\path\to\ComfyUI\custom_nodes\comfyui-blackwell-kernels
git pull
build.bat
```

Then restart ComfyUI.

---

## Disabling

To go back to stock ComfyUI without uninstalling:

1. Rename the folder:
   - Find `ComfyUI\custom_nodes\comfyui-blackwell-kernels`
   - Right-click → Rename → `comfyui-blackwell-kernels.disabled`
2. Restart ComfyUI

To re-enable, rename it back (remove `.disabled`).
