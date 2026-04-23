# Installing Pre-Built Kernels on Windows — Quick Start

This is the fastest way to install. No compilers needed — just download and run two commands.

**You need:** Windows 10/11, an RTX 5090, Python 3.12, and ComfyUI already installed and working.

---

## Step 1: Open a Command Prompt in your ComfyUI folder

1. Open File Explorer and navigate to your ComfyUI folder
   - Common locations: `C:\ComfyUI`, `C:\Users\YourName\ComfyUI`, `D:\ComfyUI`
2. Click in the address bar at the top, type `cmd`, and press Enter
   - This opens a Command Prompt already in the right folder

## Step 2: Download the custom node

Type this and press Enter:

```
git clone https://github.com/DarrellThomas/comfyui-blackwell-kernels.git custom_nodes\comfyui-blackwell-kernels
```

If you don't have `git` installed: go to https://git-scm.com/downloads/win, install it, then close and reopen the Command Prompt and try again.

## Step 3: Install the pre-built kernel

Go to the releases page and download the Windows wheel:

**https://github.com/DarrellThomas/comfyui-blackwell-kernels/releases/latest**

Click on `comfyui_blackwell_kernels-0.1.0-cp312-cp312-win_amd64.whl` to download it.

Then in your Command Prompt, install it (adjust the path to wherever you downloaded it):

```
pip install %USERPROFILE%\Downloads\comfyui_blackwell_kernels-0.1.0-cp312-cp312-win_amd64.whl
```

You should see `Successfully installed comfyui-blackwell-kernels-0.1.0`.

## Step 4: Start ComfyUI and verify

Start ComfyUI the way you normally do. Look in the console window for:

```
[Blackwell Kernels] ACTIVE — RTX 5090 Flash Attention replacing SDPA (D=40/64/128)
```

**That's it! You're done.** Your image generation is now using custom RTX 5090 kernels. All your existing workflows work exactly the same — just faster.

---

## Optional: Run the speed test

To see how much faster the kernels are on your machine:

```
cd custom_nodes\comfyui-blackwell-kernels
python benchmark.py
```

This prints a side-by-side comparison table showing stock vs custom kernel speed for different model types.

---

## Troubleshooting

**"git is not recognized"**
Install Git from https://git-scm.com/downloads/win, then close and reopen the Command Prompt.

**"pip is not recognized"**
Your Python isn't on the PATH. Try using the full path: `C:\Users\YourName\AppData\Local\Programs\Python\Python312\Scripts\pip.exe` instead of just `pip`. Or reinstall Python and check "Add Python to PATH".

**"ERROR: ... is not a supported wheel on this platform"**
This means either:
- You're not on Python 3.12 (check with `python --version`)
- You downloaded the Linux wheel instead of the Windows one (the filename should end in `win_amd64.whl`)

**"Extension not compiled" when starting ComfyUI**
The pip install didn't work. Try running the pip install command again and look for errors.

**"Skipped — GPU is CC X.Y, need >= 12.0"**
Your GPU isn't an RTX 5090. These kernels only work on RTX 5090 and future Blackwell GPUs.

**No "[Blackwell Kernels]" message at all**
The custom_nodes folder might be in the wrong place. Make sure the folder structure looks like:
```
ComfyUI\
  custom_nodes\
    comfyui-blackwell-kernels\
      __init__.py    <-- this file must exist here
```

---

## Updating

When a new version is released:

1. Download the new `.whl` from the releases page
2. Run: `pip install --force-reinstall %USERPROFILE%\Downloads\comfyui_blackwell_kernels-<new-version>-cp312-cp312-win_amd64.whl`
3. Restart ComfyUI

## Disabling

Rename the folder to turn it off without uninstalling:

1. Go to `ComfyUI\custom_nodes\`
2. Rename `comfyui-blackwell-kernels` to `comfyui-blackwell-kernels.disabled`
3. Restart ComfyUI

Rename it back to re-enable.
