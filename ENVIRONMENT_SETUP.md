This document explains how to set up a Windows server to run the 360 video processing pipeline.

## Overview

The runner:
- Polls the Media Server for tasks
- Downloads Insta360 front/back videos
- Stitches to a single MP4 using Insta360 MediaSDK
- Extracts frames + optional blur
- Runs Stella VSLAM in WSL for route calculation
- Uploads selected frames + updates metadata/status in the API

## 1) Machine requirements

- **OS**: Windows Server 2022 (or Windows 10/11)
- **GPU**: NVIDIA recommended (helps with MediaSDK + optional blur detection)
- **Disk**: enough space for:
  - work directory (downloaded INSV + stitched MP4 + extracted frames)
  - PyTorch GPU wheels (multi‑GB) if enabling GPU blur
- **Networking**: stable outbound access to the APIs + storage

Example machine: Hetzner GEX44 + Windows Server 2022.

## 2) Install prerequisites

### 2.1 Install WSL

Install WSL and a Linux distribution (Ubuntu recommended). Stella VSLAM is executed via `wsl.exe`.

### 2.2 Install Stella VSLAM in WSL

In WSL:
- Build/install Stella VSLAM binaries (and any required tools like `o2g`)
- Copy required Stella files to the expected locations:
  - vocabulary file (e.g. `orb_vocab.fbow`)
  - config YAML (e.g. `insta360_equirect.yaml`)

### 2.3 Install Insta360 MediaSDK (Windows)

Install Insta360 Windows MediaSDK (MediaSDKTest.exe). Internal link:
`https://drive.google.com/file/d/1MRNXNIoA0xfNEaqfkuFD08Fo1E-2hfbX/view?usp=drive_link`

### 2.4 Install Python 3.11 (Windows)

Install **Python 3.11 (64-bit)**.

Notes:
- This project supports other versions, but **GPU PyTorch wheels are most reliable on Python 3.10/3.11 on Windows**.
- Ensure `py --list` shows Python 3.11, and that you run the runner using the same Python/venv you install packages into.

### 2.5 Install FFmpeg (Windows)

FFmpeg is used for video probing and frame extraction.

Install options:
- **Recommended**: download a Windows build of FFmpeg, extract it, and add the `bin` folder to **PATH**.
  - Example target: `C:\ffmpeg\bin`

Verify installation:

```bat
ffmpeg -version
ffprobe -version
```

### 2.6 Install NVIDIA driver / CUDA runtime (for GPU features)

If you want GPU acceleration (MediaSDK CUDA + PyTorch CUDA for blur detection), ensure the NVIDIA driver is installed.

Verify the driver is installed and visible:

```bat
nvidia-smi
```

Notes:
- PyTorch CUDA wheels bundle the CUDA runtime, but still require a **working NVIDIA driver** on the host.
- If `torch.cuda.is_available()` is `False`, the most common causes are:
  - missing/old NVIDIA driver
  - CPU-only PyTorch build installed
  - wrong Python interpreter/venv being used

## 3) Prepare directories

Recommended layout:
- **Code directory**: `C:\insta360-prosessing\`
- **Work directory** (large disk): `C:\temp\work\`

Create folders if needed:

```bat
mkdir C:\insta360-prosessing
mkdir C:\temp\work
```

## 4) Get the code

Clone/pull the repo into the code directory (example):
- `C:\insta360-prosessing\`

## 5) Create venv + install Python deps

From `C:\insta360-prosessing\`:

```bat
py -V:3.11 -m venv venv
venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

### Optional: enable GPU blur (PyTorch + TorchVision)

GPU blur requires PyTorch + TorchVision. These downloads are **very large**.

If your `C:` temp is small, redirect temp to another drive before install:

```bat
set TMP=D:\temp
set TEMP=D:\temp
pip install -r requirements-gpu.txt
```

Verify PyTorch sees CUDA:

```bat
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 6) Configure the pipeline (`config.json`)

Edit `config.json` and set at minimum:
- **MediaSDK**
  - `mediasdk_executable`: path to `MediaSDKTest.exe`
- **Work directory**
  - `local_work_dir`: e.g. `C:\\temp\\work`
- **Media Server**
  - `media_server_api_domain`
  - `media_server_api_key`
  - `worker_id`
- **Stella VSLAM**
  - `stella_executable` (path inside WSL or mount mapping)
  - `stella_config_path`
  - `stella_vocab_path`
  - `stella_results_path`

Optional tuning:
- `candidates_per_second` / `stella_fps`
- `frame_upload_parallelism`
- `blur_settings` (including `detector_batch_size`)

## 7) Set up Task Scheduler jobs

Create Task Scheduler tasks:

1) **Run `run.bat` at login**
   - Trigger: “At log on”
   - Delay: ~30 seconds
   - Run only when user is logged on (recommended)

2) **Reboot machine nightly**
   - Trigger: daily at 01:00

3) **Lock screen after logon**
   - Trigger: “At log on”
   - Action: lock workstation

## 8) Windows settings

- Enable automatic login for the service user after reboot
- Disable forced logoff / interactive prompts that would stop scheduled tasks

## 9) Verify it runs

From an activated venv in `C:\insta360-prosessing\`:

```bat
python processing_runner.py
```

Expected behavior:
- Logs show periodic polling
- When a task exists, it processes it end-to-end

## 10) Ops checklist (things to remember)

- **Binaries in PATH**
  - `ffmpeg` and `ffprobe` available (`ffmpeg -version`)
  - `wsl.exe` available
  - `mediasdk_executable` points to `MediaSDKTest.exe`

- **Python/venv**
  - Run the service using the same venv you installed packages into:
    - `C:\insta360-prosessing\venv\Scripts\python.exe`
  - GPU blur requires Python 3.11 (recommended) + CUDA-capable torch build:
    - `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`

- **Disk / temp space**
  - Ensure `local_work_dir` has enough space for INSV + stitched MP4 + extracted frames.
  - For torch CUDA wheel installs (multi‑GB), set `TMP`/`TEMP` to a drive with space.

- **WSL path mapping**
  - Stella reads frames via `/mnt/c/...` paths; ensure Windows paths map correctly into WSL.
  - Verify Stella config + vocab paths exist inside WSL.

- **Secrets**
  - `config.json` contains API keys/tokens. Keep it out of version control and secure server copies.

- **Task Scheduler**
  - Use “Run only when user is logged on” if the workflow requires interactive session/GPU.
  - Nightly reboot + startup log cleanup is expected in this setup.

- **Important behavior**
  - The stitched MP4 is currently **not uploaded** as a binary; the pipeline creates a processed video object in the API but skips MP4 upload.

## 11) Updating the code on the server (no CI/CD yet)

There is currently **no automated git pipeline / CI/CD deployment** for the processing server.

To update code on the server:
- Pull changes from GitHub manually (e.g. `git pull`)
- Restart the runner (Task Scheduler task / reboot / restart process)
- Verify logs after restart
