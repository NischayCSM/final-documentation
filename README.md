# Efficient and Lightweight Computer Vision Models for Paddy Disease Detection

## Project Workflow

This is the end-to-end path from a fresh clone to a finished set of quantization results. Every step links to the section with the full detail — this is just the map.

```
 [1] Set up Linux/WSL2 + venv  (Section 4)
        │
        ▼
 [2] Download the Paddy Doctor dataset  (Section 3)
        │
        ▼
 [3] Train a model:  python train.py --model <name> --data_dir <path>  (Section 5)
        │
        ▼
     <model>_paddy.pth  +  confusion_matrix_<model>.png  +  console classification report
        │
        ▼
 [4] Benchmark it:  python quantize.py --model <name|all> --data_dir <path> --weights_dir <path>  (Section 6)
        │
        ▼
 [5] Compare FP32 / FP16 / PTQ / QAT results  (Section 6.3)
        │
        ▼
 [6] Read results against the known caveats before trusting any number  (Section 7)
```

1. **Set up the environment (Section 4).** Get onto Linux/WSL2 if you're on Windows, install the C/C++ build tools and CUDA Toolkit, create the venv, and install the pinned PyTorch build plus `requirements.txt`. Do this once, before anything else.
2. **Get the dataset (Section 3).** Download the Paddy Doctor dataset from Kaggle (CLI or manual), store it outside the repo, and note the path to `train_images/`.
3. **Train one or more architectures (Section 5).** Run `train.py --model <name> --data_dir <path>` once per architecture you care about. Each run writes `<model>_paddy.pth`, a confusion-matrix image, and a classification report to whatever directory you ran it from.
4. **Benchmark quantization (Section 6).** Run `quantize.py --model <name-or-all> --data_dir <path> --weights_dir <path-to-.pth-files>`. This loads the weights from step 3 and is where FP32/FP16/PTQ/QAT benchmarking happens — it does not train anything itself.
5. **Compare the four precision stages (Section 6.3).** Read the per-model results tables (size, accuracy, GPU/CPU latency) that print to the console, or redirect them to a file.
6. **Read the results with the known caveats in mind (Section 7).** In particular, don't take EfficientNet-B0's quantized weights at face value (Section 7's discussion, and the flagged result in Section 6.3) — check the known-issues list before drawing conclusions from any anomalous number.

Steps 1 and 2 are one-time setup; steps 3–6 repeat for every architecture you want to train and benchmark.

## 1. Overview

The project has two stages, run through two scripts:

1. **`train.py`** — loads the paddy disease images, splits them into training/validation sets, fine-tunes a pretrained backbone with a new classification head, trains for 10 epochs by default (configurable via `--epochs`), and saves the trained weights plus a classification report and confusion matrix.
2. **`quantize.py`** — loads the weights `train.py` produced and benchmarks them under four precision/format stages (FP32 baseline, BFloat16/FP16, INT8 Post-Training Quantization, INT8 Quantization-Aware Training), reporting model size, accuracy, and CPU/GPU latency for each.

`dataset.py` is not run directly — it's a shared module imported by both scripts so the data-loading logic (transforms, stratified split, `ImageFolder` setup) only has to be correct in one place.

**Both scripts are meant to be run inside the same Linux environment.** `quantize.py` needs a C/C++ compiler and (on WSL2) a specific GPU library path setup that native Windows can't provide cleanly — see Section 4. Rather than juggling two different operating systems and two different Python environments for one project, this guide sets everything up once, on Linux, and both `train.py` and `quantize.py` run out of that same setup. If you're on Windows, install WSL2/Ubuntu first (covered in Section 4) and do everything from inside it.

Both scripts support the same six backbone architectures, selected via `--model`:

| `--model` value | Architecture | Backbone weights |
|---|---|---|
| `resnet18` | ResNet18 | `ResNet18_Weights.DEFAULT` |
| `vgg16` | VGG16 | `VGG16_Weights.DEFAULT` |
| `densenet121` | DenseNet121 | `DenseNet121_Weights.DEFAULT` |
| `googlenet` | GoogLeNet (Inception v1) | `GoogLeNet_Weights.DEFAULT` |
| `mobilenet_v2` | MobileNetV2 | `MobileNet_V2_Weights.DEFAULT` |
| `efficientnet_b0` | EfficientNet-B0 | `EfficientNet_B0_Weights.DEFAULT` |

`quantize.py` additionally accepts `--model all` to benchmark all six in one run.

## 2. Repository Structure

```
.
├── dataset.py          # shared: data loading, transforms, train/val split
├── train.py             # trains a model (--model ...)
├── quantize.py          # benchmarks a trained model's quantization (--model ... | --model all)
├── requirements.txt      # all dependencies for both scripts — one environment, one file
└── README.md
```

Running `train.py` for a given `--model` creates, in the **current working directory** (wherever you run the command from — there's no `--output_dir` flag to redirect this):
- `<model>_paddy.pth` — trained model weights (e.g. `resnet18_paddy.pth`)
- `confusion_matrix_<model>.png` — confusion matrix heatmap
- Console output: per-epoch train/val loss and accuracy, and a final `classification_report` with real class names and the correct model name in the header

### 2.1 What `dataset.py` actually does

`dataset.py` exposes a single function, `get_dataloaders(data_dir, batch_size=32)`, which both `train.py` and `quantize.py` import and call. It:

1. **Validates `data_dir` up front** — if the path doesn't exist, it raises `FileNotFoundError` immediately, before any of the (slower) dataset-loading steps below run.
2. **Builds two separate `ImageFolder` instances over the same `data_dir`** — one with training-time augmentation (`RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter`), one with plain evaluation transforms (`Resize(256)` → `CenterCrop(224)`) — both normalized with standard ImageNet mean/std. Using two separate `ImageFolder` instances (rather than one dataset with one transform) is what keeps augmentation from leaking into the validation split.
3. **Takes class names directly from the training `ImageFolder`** (`train_full.classes`) — this is the *only* source of class labels anywhere in the pipeline; nothing reads `train.csv` (see Section 7.2).
4. **Splits 80/20 with `sklearn.model_selection.train_test_split`**, stratified by class and with a fixed `random_state=42` — so the same physical images end up in train vs. val on every run, for both `train.py` and `quantize.py`, as long as the underlying `data_dir` contents don't change.
5. **Picks `num_workers` automatically based on OS** — `0` on native Windows (`os.name == 'nt'`), `4` everywhere else. Since this project runs entirely inside Linux/WSL2 (Section 4), you'll always get `num_workers=4` in practice; the Windows branch only matters if you ever run these scripts outside the setup this README describes.
6. **Returns `(train_loader, val_loader, class_names)`** — a `DataLoader` pair plus the list of class name strings, which is what both `train.py` and `quantize.py` build their models and reports around.

### 2.2 Code Architecture / Data Flow

This is how the three files actually connect at runtime — distinct from the Project Workflow diagram above, which is about the *commands you run*; this one is about what the *code itself* does when you run them.

```
                              dataset.py
                        get_dataloaders(data_dir, batch_size)
                     (ImageFolder + augmentation + 80/20 split)
                                    │
                     ┌──────────────┴──────────────┐
                     │                              │
                     ▼                              ▼
                train.py                      quantize.py
        --model <name> --data_dir      --model <name>|all --data_dir
                     │                    --weights_dir <path>
                     │                              │
          builds backbone + new                loads <model>_paddy.pth
          classifier head, trains              from --weights_dir for
          for --epochs, evaluates               each requested model
                     │                              │
                     ▼                              ▼
        <model>_paddy.pth                 runs FP32 baseline, then
        confusion_matrix_<model>.png      FP16 (single-model pass only),
        console classification report     INT8 PTQ, INT8 QAT
                     │                              │
                     └──────────────┬───────────────┘
                                    ▼
                    <model>_paddy.pth is the ONLY link
                    between the two scripts — quantize.py
                    reads it via --weights_dir and will
                    raise FileNotFoundError if it's missing
                                    │
                                    ▼
                    quantize.py console output:
                    one Markdown results table per model
                    (size, accuracy, GPU/CPU latency per stage)
```

The two scripts never call each other and never share process memory — the trained weights file is the entire interface between Phase 1 and Phase 2. This is also why `--weights_dir` in Section 6 has to point at wherever `train.py` actually wrote its output (Section 5): there's no other channel connecting them.

### 2.3 Estimated Runtime & Hardware

There's no fixed benchmark for total training/quantization wall-clock time in this repo — it depends entirely on your GPU, CPU, and how many of the six architectures you run. What *is* fixed is each architecture's relative compute cost, since that follows directly from parameter count and FLOPs, which don't change with hardware:

| Model | Parameters (approx.) | Relative training cost per epoch | Relative quantization/benchmarking cost |
|---|---|---|---|
| MobileNetV2 | ~3.5M | Lowest | Lowest |
| GoogLeNet | ~6.6M | Low | Low |
| ResNet18 | ~11.7M | Low–Medium | Low–Medium |
| EfficientNet-B0 | ~5.3M | Medium (fewer params, but its depthwise/SE ops are less GPU-parallel-friendly than plain convolutions) | Medium |
| DenseNet121 | ~8.0M | Medium–High (dense feature concatenation is memory-bandwidth-heavy despite the modest parameter count) | Medium–High |
| VGG16 | ~138M | Highest by a wide margin | Highest by a wide margin (see Section 7.9 — its quantized `torch.compile` step is skipped specifically because of this) |

This ordering is consistent with the CPU latency figures already measured and reported in Section 6.3 (e.g. VGG16's ~1.6s baseline CPU inference vs. ResNet18's ~250ms for a single batch) — training and quantization time per epoch scale with the same underlying compute cost.

For absolute numbers on your own hardware, fill in the table below after your first run of each script (`time python train.py --model <name> --data_dir <path>` on Linux gives you real wall-clock time directly):

| Model | Training time (`train.py`, 10 epochs) | Quantization benchmarking time (`quantize.py`, all 4 stages) | GPU used |
|---|---|---|---|
| ResNet18 | | | |
| DenseNet121 | | | |
| EfficientNet-B0 | | | |
| MobileNetV2 | | | |
| GoogLeNet | | | |
| VGG16 | | | |

If you're deciding whether to run all six architectures or just one: MobileNetV2, GoogLeNet, and ResNet18 are the cheapest three to iterate on, so they're a reasonable place to start if you're constrained on time or GPU access, with VGG16 left until last given how much longer it takes per the table above.

## 3. Dataset Setup

We use the Kaggle **Paddy Doctor: Paddy Disease Classification** dataset for this project: https://www.kaggle.com/c/paddy-disease-classification/

### 3.1 Downloading the dataset

You need a (free) Kaggle account, and you need to have accepted the competition rules on the competition page above before you can download the data — otherwise both options below will fail with a 403 error. Run these from inside your Linux/WSL2 terminal (Section 4 covers getting that set up, if you haven't already).

**Option A — Kaggle CLI (recommended):**

1. Install the CLI:
   ```bash
   pip install kaggle
   ```
2. Get an API token: go to your Kaggle account settings (https://www.kaggle.com/settings) → "API" → "Create New Token". This downloads a `kaggle.json` file containing your credentials.
3. Place that file where the CLI expects it:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json   # Kaggle refuses to run if this file is group/world-readable
   ```
4. Download and unzip:
   ```bash
   kaggle competitions download -c paddy-disease-classification
   unzip paddy-disease-classification.zip -d paddy-disease-classification
   ```

**Option B — Manual download (no API token needed):**

1. Go to the competition page: https://www.kaggle.com/c/paddy-disease-classification/
2. Log in (or create a free account) and accept the competition rules if you haven't already — there's an "I Understand and Accept" prompt the first time.
3. Click the **"Data"** tab on the competition page.
4. Scroll down and click **"Download All"** (this downloads a single zip containing `train.csv`, `train_images/`, `test_images/`, and a sample submission file).
5. Unzip it wherever you plan to store it (see Section 3.2 below): `unzip paddy-disease-classification.zip -d paddy-disease-classification`.

You only strictly need `train_images/` out of what's in the zip — `train.csv`, `test_images/`, and the sample submission aren't used by anything in this repo (see Section 7.2). It's fine to keep `train.csv` around anyway since it comes bundled in the zip regardless.

### 3.2 Where to store it

Either way, you end up with a folder containing (at minimum) `train_images/` and `train.csv`. Store this **outside the repository**, not inside it — the dataset is several GB, and committing it (even accidentally) will bloat the repo and likely exceed GitHub's file size limits. Make a sibling folder next to the repo — not inside it — and call it `dataset/`, e.g.:

```
~/projects/
├── paddy-disease-classification/     ← this repo (train.py, quantize.py, dataset.py, README.md, ...)
└── dataset/
    └── paddy-disease-classification/
        ├── train_images/
        │   ├── bacterial_leaf_blight/
        │   ├── bacterial_leaf_streak/
        │   ├── ...
        │   └── tungro/
        └── train.csv
```

If you keep the dataset inside the repo folder anyway (e.g. for convenience on a personal machine), at least add it to `.gitignore` so it never gets committed:
```
train_images/
train.csv
*.zip
```

The `train_images/` folder must have one subfolder per class — this is how it ships from Kaggle already, so you shouldn't need to reorganize anything after unzipping.

Since both `train.py` and `quantize.py` run in the same Linux/WSL2 environment (Section 4), you only need **one copy** of the dataset — no need to maintain a separate Windows-path copy and a separate Linux-path copy. If you're on WSL2, this folder can live under your Linux home directory (`~/projects/...`) or be accessed from your Windows filesystem via `/mnt/c/...` — either works, but keeping it in the Linux filesystem directly is generally faster.

### 3.3 Point the scripts at it — no editing required

Nothing needs to be edited in `dataset.py`, `train.py`, or `quantize.py` to point at your dataset. Both entry-point scripts take the dataset location as a command-line argument instead:

```bash
python train.py --model resnet18 --data_dir /home/you/projects/dataset/paddy-disease-classification/train_images
python quantize.py --model resnet18 --data_dir /home/you/projects/dataset/paddy-disease-classification/train_images --weights_dir .
```

See Section 5 and Section 6 for the full command reference, including the other flags (`--weights_dir`, `--epochs`, `--batch_size`, `--lr`).

## 4. Environment Setup — install everything before running either script

Both `train.py` and `quantize.py` run inside **one Linux environment**, set up once, before you run anything. This section covers the full install: getting onto Linux at all (if you're on Windows), the system-level build/CUDA prerequisites, the Python virtual environment, and every package both scripts need. Do all of this first — don't skip ahead to Section 5 or 6 until this section is done.

### 4.1 Get onto Linux

If you're already on Linux or Mac, skip to Section 4.2.

If you're on Windows, install **Ubuntu for Windows (WSL2)** rather than trying to make everything work natively — `quantize.py` specifically needs a C/C++ compiler and GPU library paths that are much more reliable to set up on Linux than to fight through on native Windows (see Section 4.2 and Section 4.6). Either run `wsl --install` from an administrator PowerShell/Command Prompt, or install "Ubuntu" from the Microsoft Store, then do everything below from inside that Ubuntu environment.

### 4.2 Install the Linux build and CUDA prerequisites

A fresh Ubuntu/WSL install has none of this by default, and `torch.compile` (used by `quantize.py`) needs all three pieces before it can compile anything. Install these **before** creating your Python environment in Section 4.3.

1. **C++ compiler and linker (`g++`, `ld`)** — a fresh Ubuntu/WSL install doesn't include these out of the box:
   ```bash
   sudo apt update
   sudo apt upgrade -y
   sudo apt install build-essential
   ```

2. **Python development headers** — needed to build the C++ extension that binds back into Python. Install the `-dev` package matching your Python version (check with `python3 --version` first), e.g. for Python 3.12:
   ```bash
   sudo apt install python3.12-dev
   ```
   (swap `3.12` for whatever `python3 --version` actually reports on your system).

3. **CUDA Toolkit (WSL-Ubuntu build)** — even if you're on WSL2 and the actual NVIDIA driver lives on the Windows host and passes through automatically into `/usr/lib/wsl/lib`, the compiler still needs the CUDA header files (e.g. things under `-I/usr/local/cuda/include`) to compile CUDA-aware code at all. Install the official NVIDIA network repo build for WSL-Ubuntu:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   sudo apt install cuda-toolkit
   ```
   On WSL2, don't install the regular (non-WSL) Linux CUDA driver package here — WSL2 already gets the driver from Windows; you only need the toolkit/headers. On native Linux with its own GPU, install your distro's normal NVIDIA driver + CUDA Toolkit instead.

To summarize why all three matter together: `build-essential` gets you a compiler at all; the CUDA Toolkit gives that compiler the CUDA headers it needs to understand CUDA code; and (on WSL2 specifically) the library-path fix in Section 4.6 is what finally lets the compiler and the runtime linker actually *find* the real driver library (`libcuda.so`). Skipping any one of these leaves you with a different failure at a different stage of the same `torch.compile` call.

### 4.3 Create the virtual environment and install Python packages

Use **Python 3.10, 3.11, or 3.12** — as of this writing, PyTorch doesn't yet ship pre-compiled wheels for the very latest Python releases, and using one that's too new produces a misleading "no matching distribution found" error on the torch install below (covered in Section 7).

Run the following commands **in this exact order**:

```bash
# Force a supported Python version for compatibility
python3.12 -m venv venv
source venv/bin/activate

# 1. Upgrade pip first — an old pip inside a fresh venv can silently fail to
#    find recent PyTorch wheels and report "no matching distribution" even
#    though the package genuinely exists (see Section 7 if this happens to you)
pip install --upgrade pip

# 2. Install PyTorch, pinned to a version/build known to work with this
#    project (explicitly appending +cu121 avoids some indexing errors)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. Install everything else both scripts need (includes torchao, pinned —
#    see Section 7 for why the torchao version matters)
pip install -r requirements.txt
```

If you don't have an NVIDIA GPU, or the CUDA install fails, use the CPU-only build instead in step 2:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Check that your GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 4.4 First-run internet access

The first time you run `train.py` for a given `--model`, it will download that architecture's pretrained ImageNet weights. Make sure you have an internet connection.

### 4.5 WSL2: the NVIDIA library path fix (informational — `quantize.py` handles this itself)

If you're on WSL2, this section explains something `quantize.py` already does for you automatically — read it if you want to understand what's happening under the hood or if you're troubleshooting.

WSL2 automatically mounts the NVIDIA driver libraries from Windows into the Linux filesystem at `/usr/lib/wsl/lib`, so GPU passthrough works without you installing a separate Linux NVIDIA driver. The problem is that this path usually isn't in your C++ toolchain's default search locations, so when `torch.compile` (TorchInductor) compiles and links its generated C++/CUDA code, the compiler and linker can't find `libcuda.so` and similar files there — you'd see an error like `cannot find -lcuda` — even though `torch.cuda.is_available()` reports `True` and ordinary (non-compiled) CUDA operations work fine.

`quantize.py` fixes this itself, at the very top of the file, before anything else runs:

```python
import os
import sys

# Automatically configure WSL NVIDIA driver library paths for torch.compile
if "LIBRARY_PATH" not in os.environ or "/usr/lib/wsl/lib" not in os.environ["LIBRARY_PATH"]:
    os.environ["LIBRARY_PATH"] = f"/usr/lib/wsl/lib:{os.environ.get('LIBRARY_PATH', '')}"

if "LD_LIBRARY_PATH" not in os.environ or "/usr/lib/wsl/lib" not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"/usr/lib/wsl/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
```

Why this needs to set **both** `LIBRARY_PATH` and `LD_LIBRARY_PATH`, and why it has to happen this early:

- **`LIBRARY_PATH`** is read by the compiler/linker (`gcc`/`g++`/`nvcc`) at **build time**, when TorchInductor is deciding where to look for libraries to link against while compiling its generated code.
- **`LD_LIBRARY_PATH`** is read by the dynamic linker (`ld.so`) at **run time**, when that already-compiled extension is loaded and needs to resolve those same shared libraries again to actually execute.

Compiling and running are two separate steps that each do their own library lookup, so both variables are needed — fixing only one gets you past compilation and then fails at runtime, or vice versa. Setting `os.environ` at import time (before `torch` or `torch.compile` are touched anywhere else in the script) means every subprocess `torch.compile` later spawns to invoke the compiler inherits these values automatically, since child processes inherit their parent's environment.

This only helps if you've already installed the actual compiler and CUDA headers from Section 4.2 — setting the *path* to `/usr/lib/wsl/lib` doesn't do anything if there's no `g++`/CUDA Toolkit for that path to matter to yet.

### 4.6 `torch.compile`'s C++ wrapper is turned off for quantized stages (informational)

`torch._inductor.config.cpp_wrapper = True` gives the fastest FP32 baseline latency numbers, but TorchInductor's C++ code generator doesn't support quantized dtypes like `quint8`. Leaving it enabled during the PTQ/QAT stages crashes with a missing-symbol error (`aoti_torch_dtype_quint8 was not declared`) rather than a clear "unsupported" message.

`quantize.py` handles this by toggling the flag per stage internally: `cpp_wrapper = True` for the FP32 baseline, and `cpp_wrapper = False` right before running PTQ or QAT (this doesn't disable `torch.compile` — it just switches to the Python-level wrapper, which does understand INT8 tensors, for those two stages). You don't need to do anything for this yourself; it's mentioned here so that if you ever modify the script and reintroduce a single global `cpp_wrapper = True`, you'll know why PTQ/QAT suddenly starts crashing.

## 5. Running Training (`train.py`)

Once Section 4 is done, `train.py` takes the architecture and dataset location as required arguments (`--model` and `--data_dir`), plus optional `--epochs` (default 10), `--batch_size` (default 32), and `--lr` (default `1e-4`) — run it once per architecture you want to train, from whichever directory you want the outputs written to:

```bash
python train.py --model resnet18        --data_dir <path-to-train_images>
python train.py --model vgg16            --data_dir <path-to-train_images>
python train.py --model densenet121      --data_dir <path-to-train_images>
python train.py --model googlenet        --data_dir <path-to-train_images>
python train.py --model mobilenet_v2     --data_dir <path-to-train_images>
python train.py --model efficientnet_b0  --data_dir <path-to-train_images>
```

There's no `--output_dir` flag — `<model>_paddy.pth` and `confusion_matrix_<model>.png` are always written to whatever directory you ran the command from. If you want the outputs somewhere specific, `cd` there first (e.g. the repo root, so it lines up with how `quantize.py` looks for weights via `--weights_dir` below). Run these one at a time — you don't have to train all six if you only care about a subset.

## 6. Running Quantization (`quantize.py`)

`quantize.py` does not train anything — it loads the weights `train.py` already produced and tests how well each model works under four precision/size formats:

| Stage | What it does |
|---|---|
| **Baseline (FP32)** | The trained model as-is, no quantization — just converted to a format that can be fairly compared against the other stages. |
| **BFloat16 / FP16** | Casts weights and activations to 16-bit, trading a small amount of precision for lower memory use and (on GPU) faster inference. |
| **INT8 Post-Training Quantization (PTQ)** | Takes an already-trained model and converts it to INT8 using a small slice of the training data for calibration — no further training involved. |
| **INT8 Quantization-Aware Training (QAT)** | Same INT8 target as PTQ, but fine-tunes the model briefly while simulating quantization noise first, which generally recovers more accuracy than PTQ. |

### 6.1 Workflow: run training first, then point `quantize.py` at the weights

`quantize.py` doesn't train anything itself — it only evaluates `.pth` files that `train.py` already produced. Follow this order:

1. **Run `train.py` first** (Section 5) for however many of the six architectures you want to benchmark — you don't have to train all six if you only care about one.
2. **Point `--weights_dir` at the folder containing those `.pth` files** when you run `quantize.py` — since `train.py` always writes its `.pth` files to whatever directory you ran it from (Section 5), this is just wherever that happened to be.
3. **If a required `.pth` file is missing from `--weights_dir`, `quantize.py` stops immediately with a `FileNotFoundError`.** This is intentional — it exists specifically so you can't accidentally benchmark and report accuracy numbers for the wrong weights without noticing. If you hit this error, double-check `--weights_dir` points at wherever `train.py` actually wrote the file, and that the filename matches the `--model` you're benchmarking.

### 6.2 Running just one model, or all six

Both are just `--model` values on the same script:

```bash
# One model
python quantize.py --model resnet18 --data_dir <path-to-train_images> --weights_dir <path-to-.pth-files>

# All six, one after another, one results table per model
python quantize.py --model all --data_dir <path-to-train_images> --weights_dir <path-to-.pth-files>
```

The extra BFloat16/FP16 stage (see the stage table above) runs for whichever model you selected — it isn't restricted to ResNet18.

`--data_dir` works exactly like it does for `train.py` (Section 3.3) — point it at your `train_images/` folder. The number of classes is inferred from however many subfolders `--data_dir` contains, the same as `train.py` — no hardcoded `num_classes` to edit. If the dataset you point at doesn't have the expected 10 paddy-disease classes, `quantize.py` will size the classifier head to match whatever it finds, but that will only load correctly against a `.pth` file trained on the same number of classes — a mismatch here will surface as a size-mismatch error when the weights are loaded, not a silent bug.

Results print directly to the console as one Markdown-formatted table per model. If you want to keep them, redirect the output to a file:
```bash
python quantize.py --model resnet18 --data_dir <path-to-train_images> --weights_dir . > resnet_quant_results.txt
```

### 6.3 Example Output

These are real results from one run, included so you have something to sanity-check your own numbers against — expect your own numbers to vary somewhat depending on hardware, PyTorch/torchao versions, and how well your `.pth` weights were trained.

```
### ResNet18
| Method                       | Model Size | Accuracy | GPU Latency (ms)    | CPU Latency (ms)    |
|-------------------------------|------------|----------|----------------------|----------------------|
| Baseline (no quantization)    | 44.90 MB   | 95.77%   | 12.15 ± 1.15 ms      | 253.65 ± 13.13 ms    |
| Post Training Quantization    | 11.39 MB   | 95.82%   | 11.75 ± 1.03 ms      | 147.39 ± 9.28 ms     |
| Quantization Aware Training   | 11.39 MB   | 94.42%   | 11.68 ± 1.04 ms      | 142.71 ± 7.92 ms     |

### DenseNet121
| Method                       | Model Size | Accuracy | GPU Latency (ms)    | CPU Latency (ms)    |
|-------------------------------|------------|----------|----------------------|----------------------|
| Baseline (no quantization)    | 29.08 MB   | 96.06%   | 41.04 ± 0.40 ms      | 971.96 ± 51.60 ms    |
| Post Training Quantization    | 8.47 MB    | 95.05%   | 41.37 ± 0.42 ms      | 281.14 ± 10.96 ms    |
| Quantization Aware Training   | 8.47 MB    | 93.99%   | 42.15 ± 1.50 ms      | 298.35 ± 19.99 ms    |

### EfficientNet_B0
| Method                       | Model Size | Accuracy | GPU Latency (ms)    | CPU Latency (ms)    |
|-------------------------------|------------|----------|----------------------|----------------------|
| Baseline (no quantization)    | 16.83 MB   | 95.29%   | 15.26 ± 0.57 ms      | 151.25 ± 10.00 ms    |
| Post Training Quantization    | 5.24 MB    | 69.13%   | 15.55 ± 0.57 ms      | 176.30 ± 10.16 ms    |
| Quantization Aware Training   | 5.24 MB    | 23.27%   | 15.34 ± 0.37 ms      | 179.09 ± 4.13 ms     |

### MobileNet_V2
| Method                       | Model Size | Accuracy | GPU Latency (ms)    | CPU Latency (ms)    |
|-------------------------------|------------|----------|----------------------|----------------------|
| Baseline (no quantization)    | 9.48 MB    | 92.64%   | 13.19 ± 0.55 ms      | 264.16 ± 14.31 ms    |
| Post Training Quantization    | 2.92 MB    | 89.76%   | 13.37 ± 0.51 ms      | 64.05 ± 2.78 ms      |
| Quantization Aware Training   | 2.92 MB    | 80.96%   | 13.18 ± 0.42 ms      | 62.61 ± 3.40 ms      |

### GoogLeNet
| Method                       | Model Size | Accuracy | GPU Latency (ms)    | CPU Latency (ms)    |
|-------------------------------|------------|----------|----------------------|----------------------|
| Baseline (no quantization)    | 22.94 MB   | 93.51%   | 13.86 ± 0.68 ms      | 236.39 ± 8.22 ms     |
| Post Training Quantization    | 6.13 MB    | 93.22%   | 13.93 ± 0.80 ms      | 137.25 ± 7.66 ms     |
| Quantization Aware Training   | 6.13 MB    | 92.45%   | 13.95 ± 0.91 ms      | 139.56 ± 6.95 ms     |

### VGG16
| Method                       | Model Size | Accuracy | GPU Latency (ms)    | CPU Latency (ms)    |
|-------------------------------|------------|----------|----------------------|----------------------|
| Baseline (no quantization)    | 537.28 MB  | 94.66%   | 81.04 ± 0.41 ms      | 1619.35 ± 32.14 ms   |
| Post Training Quantization    | 134.63 MB  | 94.90%   | 80.79 ± 0.39 ms      | 849.39 ± 10.41 ms    |
| Quantization Aware Training   | 134.63 MB  | 90.62%   | 80.74 ± 0.33 ms      | 847.85 ± 12.95 ms    |
```

`python quantize.py --model resnet18 ...` — single-model pass (includes the extra BFloat16/FP16 stage):

```
### ResNet18
| Method                       | Model Size | Accuracy | GPU Latency (ms)    | CPU Latency (ms)    |
|-------------------------------|------------|----------|----------------------|----------------------|
| Baseline (FP32)               | 44.90 MB   | 95.77%   | 11.72 ± 0.98 ms      | 250.55 ± 10.66 ms    |
| Half-Precision (FP16)         | 22.52 MB   | 95.77%   | 5.71 ± 0.78 ms       | 5345.78 ± 140.39 ms  |
| Post Training Quant. (INT8)   | 11.39 MB   | 95.62%   | 11.57 ± 0.98 ms      | 145.34 ± 7.18 ms     |
| Quant. Aware Training (INT8)  | 11.39 MB   | 93.08%   | 11.66 ± 0.69 ms      | 141.42 ± 7.83 ms     |
```

A few things worth noting from these actual numbers:

- **EfficientNet-B0's INT8 rows are a red flag, not a config typo.** The accuracy of EfficientNet-B0 goes down from 95.29% (FP32) to 69.13% (PTQ) and 23.27% (QAT). This means the FX quantization pipeline is not working well with EfficientNet-B0's architecture. Do not use EfficientNet-B0's quantized weights from this run for anything. This needs to be investigated.
- **The FP16 CPU latency (5345.78 ms) is real and expected, not a bug.** Consumer CPUs do not have native FP16 arithmetic units like GPUs do, so `torch.compile` has to emulate FP16 math on CPU, which is much slower than the FP32 baseline. The FP16 GPU latency (5.71 ms) is where the real speedup shows up. Do not think the CPU FP16 row is an error.
- ResNet18, DenseNet121, GoogLeNet, and MobileNetV2 all work well under INT8 quantization (accuracy loss is less than 2 points). This is what we expect from these models.
- **VGG16's `torch.compile` latency step is intentionally skipped** — its INT8 computational graph is large enough that building it in C++ can crash the WSL container, so `quantize.py` bypasses that step specifically for VGG16 (see Section 7), while still reporting its size and accuracy normally. Expect its GPU/CPU latency columns to read differently (or show a placeholder) compared to the other five models, even though its size and accuracy numbers remain directly comparable.

## 7. Known Issues & Fixes

### 7.1 Out-of-memory during training

Reduce the batch size, or close other GPU processes.

### 7.2 `train.csv` isn't used by anything in this repo

Unlike the wording in Section 3.1 might suggest, `dataset.py` never opens `train.csv` — it builds `class_names` and the train/val split entirely from `torchvision.datasets.ImageFolder` reading the subfolder names under `train_images/`. `train.csv` is only present because it's bundled in the Kaggle zip; you can delete it after extracting and both `train.py` and `quantize.py` will run identically. It's mentioned in Section 3 mainly so you know it's there and safe to ignore, not because anything depends on it.

### 7.3 `torch.compile` requires a C/C++ compiler

`quantize.py` uses `torch.compile`, which generates and compiles code on the fly using TorchInductor. **This is not a theoretical risk** — it's the reason Section 4 has you set up the full Linux build toolchain before running anything. If you do not want to deal with this, you can remove the `torch.compile(...)` calls and call the model directly — you will lose the compiled-latency numbers, but everything else (size, accuracy) still works.

### 7.4 FX quantization backend is x86-only (`fbgemm`)

`quantize.py` uses `get_default_qconfig("fbgemm")` / `get_default_qat_qconfig("fbgemm")` internally. `fbgemm` is the x86/AMD64 CPU quantization backend. **On ARM CPUs (e.g. Apple Silicon Macs, ARM-based Linux/Windows machines) this will fail or silently produce a non-functional quantized model.** If you're on ARM, this needs to be changed to `"qnnpack"` in the script.

### 7.5 GPU quantization silently does nothing without CUDA

The GPU stages are skipped entirely on a CPU-only machine. This is not a bug. The results table will show `N/A` in the GPU Latency column for every row on such a machine. Do not think the quantization pipeline is broken; it just means no CUDA GPU was detected.

### 7.6 Calibration uses a fixed 512-sample subset

`quantize.py` uses a fixed 512-sample calibration subset for PTQ/QAT. If your training split has fewer than 512 images total, this will raise an `IndexError`. If you're working with a much smaller dataset than the full Paddy Doctor set, this would need reducing to `min(512, len(train_dataset))`.

### 7.7 `drop_last=True` on the calibration and test loaders

This means the **last partial batch of your validation set is silently dropped** from both calibration and the reported accuracy. The accuracy numbers are computed over a slightly smaller validation set than `train.py`'s own evaluation step used.

### 7.8 `torchao` / PyTorch version compatibility

`torchao`'s quantization API and FX graph-mode quantization evolve quickly between PyTorch releases — this is why Section 4.3 pins exact versions (`torch==2.5.1+cu121`, `torchao==0.5.0`) rather than using open-ended version ranges. **Do not use `torchao>=0.6.0` with PyTorch 2.5.1** — newer `torchao` releases look for sub-byte datatypes (like `torch.int1`) that PyTorch 2.5.1 doesn't have, and this will crash at runtime. If you hit an `ImportError` or `AttributeError` inside `torchao` or `torch.ao.quantization`, check that your installed `torch` and `torchao` versions were released around the same time before assuming your code is wrong.

### 7.9 VGG16 skips the `torch.compile` latency step (by design, to prevent WSL crashes)

VGG16's INT8 computational graph is large enough that letting `torch.compile` build it in C++ could crash the entire WSL container. `measure_latency_safe()` detects `vgg16` and bypasses that step specifically for it, while still reporting size and accuracy normally. This is intentional risk management, not a missing feature — don't be surprised if VGG16's latency numbers look different in shape from the other five models' results in Section 6.3.

### 7.10 `pip install torch ...` fails with "Could not find a version that satisfies the requirement torch (from versions: none)"

This looks like torch doesn't exist for your platform, but inside a fresh WSL/Ubuntu venv it's almost always one of these instead:

- **Python version too new.** As noted in Section 4.3, PyTorch doesn't yet have pre-compiled wheels for every very-recent Python release. Using an unsupported Python version (e.g. 3.14 at the time this was written) will produce exactly this error even though nothing else is wrong. Force a supported version explicitly, e.g. `python3.12 -m venv venv`.
- **Stale `pip` inside the new venv.** A freshly created venv can ship a `pip` too old to understand the wheel-tag format PyTorch's index uses, so it silently reports zero available versions instead of a clear error. Fix: `pip install --upgrade pip`, then retry the torch install (this is already the first step in Section 4.3's install block above).
- **No real network reachability to `download.pytorch.org` from inside WSL2.** `--index-url` replaces the default PyPI index entirely, so if that specific host is unreachable (corporate proxy, VPN, or a WSL2 networking quirk), pip gets back an empty page and reports it the same way as "package doesn't exist." Test connectivity directly: `curl -I https://download.pytorch.org/whl/cu121` — if that hangs or errors, the problem is network access from WSL2, not the package or your command.
- **CUDA version mismatch with what's actually installed.** If you copy-pasted a `cu121`/`cu124`/etc. URL from somewhere without checking it against your actual CUDA setup, double-check the exact command for your hardware at https://pytorch.org/get-started/locally/ rather than assuming the one in this README's examples is current for you.

If none of these fix it, install the CPU-only build instead to unblock yourself (you'll lose GPU benchmarking numbers in Section 6.3, but everything else still runs):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 8. Disclosure of AI Tool Usage

We disclose the use of AI tools as required by the assignment. **Claude (Anthropic) and Gemini (Google) were used as coding assistants** throughout this project — including drafting `dataset.py`, `train.py`, and `quantize.py`, and implementing the safety features described above (strict weights validation, automatic WSL2 environment setup, VGG16 OOM protection, correct model-name/class-name labeling in reports). We also used them to write this README. All dataset handling, model choices, hyperparameters, and final code were reviewed and run by the author. The AI tools were only used as a coding aid, not as an autonomous agent.