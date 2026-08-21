# Paddy Disease Classification Using PyTorch: Baseline Training and Quantization Benchmarking

## Part A. Phase 1: Baseline CNN Training

## 1. Overview

There are six scripts for this project. Each script does the following things:

1. It loads the images of paddy disease.
2. It divides the data into training and validation sets.
3. It changes a pre-trained model with a new classification head.
4. It trains the model for 10 epochs.
5. It saves the trained model and a classification report.

| Script | Architecture | Backbone weights |
|---|---|---|
| `resnet.py` | ResNet18 | `ResNet18_Weights.DEFAULT` |
| `vgg.py` | VGG16 | `VGG16_Weights.DEFAULT` |
| `densenet.py` | DenseNet121 | `DenseNet121_Weights.DEFAULT` |
| `googlenet.py` | GoogLeNet (Inception v1) | `GoogLeNet_Weights.DEFAULT` |
| `mobilenet.py` | MobileNetV2 | `MobileNet_V2_Weights.DEFAULT` |
| `efficientnet.py` | EfficientNet-B0 | `EfficientNet_B0_Weights.DEFAULT` |

All six scripts follow the same process. The difference is in the model architecture and the classification head.

## 2. Repository Structure

```
├── models/
│   ├── resnet.py          # Phase 1: trains ResNet18
│   ├── vgg.py               # Phase 1: trains VGG16
│   ├── densenet.py          # Phase 1: trains DenseNet121
│   ├── googlenet.py         # Phase 1: trains GoogLeNet
│   ├── mobilenet.py         # Phase 1: trains MobileNetV2
│   └── efficientnet.py      # Phase 1: trains EfficientNet-B0
├── quantization/
│   ├── resnet.py            # Phase 2: quantization benchmark for ResNet18
│   └── all_models.py        # Phase 2: quantization benchmark for all six architectures
├── requirements-models.txt      # Phase 1 dependencies (install into venv-models)
├── requirements-quantization.txt # Phase 2 dependencies (install into venv-quant)
├── requirements.txt              # optional: combined list, both phases in one env
└── README.md
```

Note that there are two different `resnet.py` files in this repository. One is for training. The other is for quantization benchmarking.

Running each script creates the following files:

- `<model>_paddy.pth`. Trained model weights
- `confusion_matrix_<model>.png`. Confusion matrix heatmap
- Console output: per-epoch train/val loss and accuracy, final `classification_report`

## 3. Dataset Setup

We use the Kaggle **Paddy Doctor: Paddy Disease Classification** dataset for this project: https://www.kaggle.com/c/paddy-disease-classification/

### 3.1 Downloading the dataset

You need a (free) Kaggle account, and you need to have accepted the competition rules on the competition page above before you can download the data — otherwise both options below will fail with a 403 error.

**Option A — Kaggle CLI (recommended):**

1. Install the CLI:
   ```
   pip install kaggle
   ```
2. Get an API token: go to your Kaggle account settings (https://www.kaggle.com/settings) → "API" → "Create New Token". This downloads a `kaggle.json` file containing your credentials.
3. Place that file where the CLI expects it:
   - Linux/Mac/WSL: `~/.kaggle/kaggle.json`
   - Windows (native): `C:\Users\<you>\.kaggle\kaggle.json`

   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json   # Linux/Mac/WSL only — Kaggle refuses to run if this file is group/world-readable
   
4. Download and unzip:
   kaggle competitions download -c paddy-disease-classification
   unzip paddy-disease-classification.zip -d paddy-disease-classification
   

**Option B — Manual download (no API token needed):**

1. Go to the competition page: https://www.kaggle.com/c/paddy-disease-classification/
2. Log in (or create a free account) and accept the competition rules if you haven't already — there's an "I Understand and Accept" prompt the first time.
3. Click the **"Data"** tab on the competition page.
4. Scroll down and click **"Download All"** (this downloads a single zip containing `train.csv`, `train_images/`, `test_images/`, and a sample submission file).
5. Unzip it wherever you plan to store it (see Section 3.2 below) — on Windows, right-click the `.zip` → "Extract All"; on Linux/Mac, `unzip paddy-disease-classification.zip -d paddy-disease-classification`.

You only strictly need `train_images/` and `train.csv` out of what's in the zip for Phase 1 and Phase 2 — `test_images/` and the sample submission aren't used by any script in this repo yet (see the Phase 3 roadmap in Section 8).

### 3.2 Where to store it

Either way, you end up with a folder containing (at minimum) `train_images/` and `train.csv`. Store this **outside the repository**, not inside it — the dataset is several GB, and committing it (even accidentally) will bloat the repo and likely exceed GitHub's file size limits. Make a sibling folder next to the repo — not inside it — and call it `dataset/`, e.g.:

```
~/projects/
├── paddy-disease-classification/     ← this repo (models/, quantization/, README.md, ...)
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

**Remember you're working across two systems, per Section 4** — Phase 1 (`models/`) on native Windows, Phase 2 (`quantization/`) on Linux/WSL2. That means you'll likely end up with **two separate copies of the dataset** (or one copy accessed via two different path styles), one per system:
- **Windows, for Phase 1:** e.g. `C:\Users\you\projects\dataset\paddy-disease-classification\train_images`
- **Linux/WSL2, for Phase 2:** e.g. `/home/you/projects\dataset/paddy-disease-classification/train_images` — if you're using WSL2, this can be the *same* physical files as your Windows copy via the `/mnt/c/...` path (see Section B.2), so you don't necessarily need to download it twice.

### 3.3 Point the scripts at it

Every script (`models/*.py` and `quantization/*.py`) hardcodes its own `data_dir` (and, in the Phase 1 scripts, `csv_path`) at the top of the file. There's no shared config — you need to update the path in **each script individually** to wherever you stored the dataset in Section 3.2 above, matching the path style for whichever system that script actually runs on:

# models/*.py — running on native Windows
```
data_dir = r"C:\Users\you\projects\dataset\paddy-disease-classification\train_images"

# quantization/*.py — running on Linux/WSL2
data_dir = "/home/you/projects/dataset/paddy-disease-classification/train_images"
```

See Section 6.2 and Section B.2 for the phase-specific notes on this.

## 4. Environment Setup

We suggest using Python 3.10 or later.

**Use two separate environments, on two different systems.** This repo is split so that Phase 1 (`models/`) and Phase 2 (`quantization/`) don't need to share a Python environment — or even a machine:

- **Phase 1 (`models/`) → native Windows is fine.** None of the training scripts need `torch.compile` or a C/C++ compiler, so there's no reason to fight with WSL for this phase. Set up a dedicated venv for it, e.g. `venv-models`, right here on Windows and install `requirements-models.txt` into it.
- **Phase 2 (`quantization/`) → use Linux (WSL2/Ubuntu).** `torch.compile` needs a C/C++ toolchain and, on WSL2, the NVIDIA library path fixes covered in Section B.4. Set up a **second, separate** venv for it, e.g. `venv-quant`, inside your Ubuntu/WSL2 environment, and install `requirements-quantization.txt` into that one instead.

Keeping them apart means a Phase 2 `torchao`/CUDA-toolkit version bump can never break your Phase 1 training environment, and you're not stuck fighting compiler setup just to run the Phase 1 scripts. If you'd rather skip the split and use a single environment for everything (e.g. you're already set up on Linux/WSL2), `requirements.txt` in the repo root is the combined list — it's just the union of `requirements-models.txt` and `requirements-quantization.txt`, nothing extra.

This section (4) covers the Phase 1 / Windows environment; jump to Section B.4 when you're ready to set up the Phase 2 / Linux environment.

```
python -m venv venv-models
venv-models\Scripts\activate      # Windows PowerShell/cmd; on Linux/Mac: source venv-models/bin/activate
pip install -r requirements-models.txt
```

### 4.1 Installing PyTorch

Install PyTorch before installing the rest of the requirements.

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the rest:

```
pip install -r requirements-models.txt
```

Check that your GPU is visible before training:

```
python -c "import torch; print(torch.cuda.is_available())"
```

### 4.2 First-run internet access

The first time each script runs it will download the pre-trained ImageNet weights for the model. Make sure you have an internet connection.

## 5. Running the Scripts

Each script is standalone and self-contained:

```
python models/resnet.py
python models/vgg.py
python models/densenet.py
python models/googlenet.py
python models/mobilenet.py
python models/efficientnet.py
```

Run them one at a time.

## 6. Known Issues & Fixes

### 6.1 `densenet.py`. CUDA check is not working

Fix the CUDA check in `densenet.py`:

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 6.2 Hardcoded Windows paths

Update the dataset paths in each script.

### 6.3 Out-of-memory on GPU

Reduce the batch size. Close other GPU processes.

### 6.4 `num_workers=0` in all `DataLoader`s

Set `num_workers` to 0 to avoid problems on Windows.

### 6.5 GoogLeNet classifiers

Keep the branch for auxiliary classifiers in `googlenet.py`.

### 6.6 The labels on the confusion matrix and report are copied from elsewhere

The helper function `generate_report_and_matrix` in the files `mobilenet.py`, `vgg.py`, `resnet.py`, and `efficientnet.py` all print the header `CLASSIFICATION REPORT (MobileNetV2)`. This is the same for all of them even though they are different models. This is a display issue but it does not affect the actual numbers. You should not be confused by the label when you read the output on the console for a model like `vgg.py`.

### 6.7 The `train.csv` file is not used

All the scripts load the `train.csv` file and print some information about it. They do this just to check that the dataset was loaded correctly. The actual labels for training come from the names of the subfolders in the image folder, not from the `train.csv` file. If the path to the `train.csv` file is wrong but the path to the image folder is right, the training will still work. If the path to the image folder is wrong (even if the path to `train.csv` is right), the training will fail immediately.

---

## Part B. Phase 2: Testing How Well The Models Work With Less Precision

Phase 2 does not train the models again. It uses the weights from Phase 1 and tests how well each model works under different precision/size formats. It tests four ways of running the same model:

| Stage | What it does |
|---|---|
| **Baseline (FP32)** | This is the model from Phase 1 without any changes. It is just converted to a format that can be compared against the other stages. |
| **BFloat16** | This is for the ResNet18 model. It changes the weights and the data to use less memory. This is a way to make the model use less memory without losing much accuracy. |
| **INT8 Post-Training Quantization (PTQ)** | This takes a model that has already been trained and converts it to use INT8 data. It does this by looking at a part of the training data for calibration. |
| **INT8 Quantization-Aware Training (QAT)** | This is similar to PTQ. It also trains the model a little bit more to make it work better at INT8. |

There are two scripts:

- **`quantization/all_models.py`**. This script tests all six models from Phase 1.
- **`quantization/resnet.py`**. This script tests the ResNet18 model in more detail.

### B.1 Workflow: run Phase 1 first, then place the weights alongside Phase 2

Phase 2 doesn't train anything itself — it only evaluates weights that Phase 1 already produced. Follow this order:

1. **Run the Phase 1 scripts first.** Each script in `models/` (see Section 5) saves a `.pth` weights file when it finishes, e.g. `models/resnet.py` produces `resnet_paddy.pth`. Do this for however many of the six architectures you want to benchmark in Phase 2 — you don't have to train all six if you only care about one.
2. **Copy (or move) each `.pth` file into the `quantization/` folder**, i.e. the same folder as `quantization/resnet.py` and `quantization/all_models.py`. Both scripts look for the weights files (`resnet_paddy.pth`, `densenet_paddy.pth`, `efficientnet_paddy.pth`, `mobilenet_paddy.pth`, `googlenet_paddy.pth`, `vgg_paddy.pth`) in their own working directory — not in `models/` — so if you run `python quantization/resnet.py` from the repo root without moving the file, it won't find `resnet_paddy.pth` and will silently fall back to an un-fine-tuned ImageNet model (see the warning behavior described below).
3. If a `.pth` file is missing from `quantization/`, the script will use a default ImageNet weight instead. The accuracy with the default weight will not be as good as with the one you trained — this isn't a crash, but you should check the console for the "falling back to default weights" warning before trusting the accuracy numbers.

### B.1.1 Running just one model instead of all six

`quantization/all_models.py` loops over all six architectures. If you only want to benchmark a single model, use `quantization/resnet.py` as your starting point instead — despite the name, you don't have to leave it as ResNet18. All you need to change are **two things**:

1. **The weights path** — swap `weights_path = "resnet_paddy.pth"` for whichever `.pth` file you copied into `quantization/` in Section B.1 (e.g. `weights_path = "vgg_paddy.pth"`).
2. **The model-building function** — swap the body of `get_resnet18()` for the matching architecture's construction code.

The six model-construction snippets below are the exact ones `quantization/all_models.py` already uses internally, inside its `get_model_standard()` factory function — copy the branch for whichever architecture you want directly into `quantization/resnet.py`'s `get_resnet18()` (or rename the function to match, it doesn't have to stay called `get_resnet18`):

### ResNet18  → weights_path = "resnet_paddy.pth"
```
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

### DenseNet121  → weights_path = "densenet_paddy.pth"
```
model = models.densenet121(weights="DEFAULT")
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
```

### EfficientNet-B0  → weights_path = "efficientnet_paddy.pth"
```
model = models.efficientnet_b0(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
```

### MobileNetV2  → weights_path = "mobilenet_paddy.pth"
```
model = models.mobilenet_v2(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
```

### VGG16  → weights_path = "vgg_paddy.pth"
```
model = models.vgg16(weights="DEFAULT")
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
```

### GoogLeNet  → weights_path = "googlenet_paddy.pth"
```
model = models.googlenet(weights="DEFAULT")
model.aux_logits = False
model.fc = nn.Linear(model.fc.in_features, num_classes)
```


After building `model` with whichever branch you picked, keep the loop that follows it in `get_model_standard()` in `quantization/all_models.py` — it disables `inplace` operations across the whole model, which the quantization/tracing steps later in the script depend on:

```
for m in model.modules():
    if hasattr(m, 'inplace'):
        m.inplace = False
```

If you find yourself doing this often, it's simpler to just run `quantization/all_models.py` directly and read off the one row you care about from its per-model results tables (see Section B.5.1) — the swap above is mainly useful if you specifically want the extra BFloat16/FP16 stage that only `quantization/resnet.py` has (see the Phase 2 stage table above), applied to an architecture other than ResNet18.

### B.2 The Path To The Dataset

You need to tell the script where the dataset is. The path is in the `data_dir` variable. You need to change this to the path where your dataset lives.

### B.3 The Number Of Classes

The script assumes that there are 10 classes in the dataset. If you have a different number of classes you need to change the `num_classes` variable.

### B.4 Things You Need To Install
You need to install `torchao` to run these scripts. You also need to have a C++ compiler installed. 

Use a Linux system for Phase 2 — and a second, separate environment from Phase 1. As introduced in Section 4, this repo is meant to be run across two environments: Phase 1 (`models/`) on native Windows in its own `venv-models`, and Phase 2 (`quantization/`) on Linux in its own `venv-quant`. Given the C++ compiler requirement in Section B.4.1 and the GPU library path issue below, quantization benchmarking is much more reliable on Linux than on native Windows. If you're on Windows, install Ubuntu for Windows (WSL2) rather than fighting with the MSVC Build Tools path — either run `wsl --install` from an administrator PowerShell/Command Prompt, or install "Ubuntu" from the Microsoft Store, then set it up and run everything from inside that Ubuntu environment.

Create the Phase 2 virtual environment inside that Linux/WSL2 environment. **Crucially, you must use Python 3.12 (or 3.11/3.10).** PyTorch 2.5.1 does not yet have pre-compiled wheels for Python 3.14, and using 3.14 will result in a "no matching distribution found" error. 

Keeping the two venvs separate means a Phase 2 `torchao`/CUDA-toolkit version bump can never break your Phase 1 training environment, and stops Phase 2 dependencies from colliding with any other PyTorch project. `venv-quant` needs PyTorch installed into it separately; it does not inherit anything from `venv-models`. 

Run the following commands in this exact order:

# Force Python 3.12 for compatibility
```
python3.12 -m venv venv-quant
source venv-quant/bin/activate    
```

# 1. Upgrade pip first
```
pip install --upgrade pip
```

# 2. Install PyTorch 2.5.1 (Explicitly appending +cu121 to avoid indexing errors)
```
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

# Install the libraries from the .txt file
```
pip install -r requirements-quantization.txt
```


### B.4.1 Install the Linux build and CUDA prerequisites first
A fresh Ubuntu/WSL install has none of this by default, and `torch.compile` needs all three pieces before it can compile anything. Install these before touching the export commands in Section B.4.2 — otherwise the exports have nothing correctly set up to point at, and you'll just trade one error for another.

**C++ compiler and linker (g++, ld)** — a fresh Ubuntu/WSL install doesn't include these out of the box:

```
sudo apt update
sudo apt upgrade -y
sudo apt install build-essential
```


**Python development headers** — `torch.compile` builds a C++ extension that binds back into Python, which needs the Python C-API headers (`Python.h`). Install the `-dev` package matching your Python version:

```
sudo apt install python3.12-dev
```

**CUDA Toolkit (WSL-Ubuntu build)** — even though the actual NVIDIA driver lives on the Windows host and passes through automatically into `/usr/lib/wsl/lib`, the compiler still needs the CUDA header files (e.g., things under `-I/usr/local/cuda/include`) to compile CUDA-aware code at all. Install the official NVIDIA network repo build for WSL-Ubuntu:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit
```

*Note: Don't install the regular (non-WSL) Linux CUDA driver package here — WSL2 already gets the driver from Windows; you only need the toolkit/headers.*

### B.4.2 WSL2: fix the NVIDIA library path before running quantization
If you're on WSL2 with an NVIDIA GPU, do this before running either quantization script, or `torch.compile` will fail to find the CUDA driver libraries (`cannot find -lcuda`) during compilation.

WSL2 automatically mounts the NVIDIA driver libraries from Windows into the Linux filesystem at `/usr/lib/wsl/lib`. The problem is that this path usually isn't in your C++ toolchain's default search locations. 

Run these before running your Python script:

```
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

*Tip: If this fixes the issue, add both export lines to the bottom of your `~/.bashrc` (or `~/.zshrc`) so they're set automatically in every new terminal session.*

### B.4.3 Crucial PyTorch Compiler Configuration
PyTorch's highly experimental C++ wrapper (`config.cpp_wrapper = True`) provides maximum speed for FP32 baselines, but the C++ code generator **does not yet support quantized data types** (like `quint8`). If left enabled during quantization, the compiler will crash with a missing scope error (`aoti_torch_dtype_quint8 was not declared`).

To fix this, you must dynamically toggle the wrapper in your scripts:
*   Use `config.cpp_wrapper = True` for the **FP32 Baseline**.
*   Set `config.cpp_wrapper = False` right before running **PTQ** or **QAT**. (This does *not* turn off the compiler; it simply uses the Python-level wrapper which safely understands INT8 tensors).

### B.5 Running The Scripts
Once your environment is properly configured, navigate into the `quantization` directory and run the scripts using `python3`:

```
cd quantization
python3 all_models.py
python3 resnet.py
```

These scripts will print the benchmarking results directly to the console. If you want to save the results, you can redirect the output to a text file (e.g., `python3 resnet.py > results.txt`).

### B.5.1 Example Output

Here is an example of what the output might look like:

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


**`quantization/resnet.py`** — ResNet18 pass (includes the half/16-bit precision stage):


### ResNet18
| Method                       | Model Size | Accuracy | GPU Latency (ms)    | CPU Latency (ms)    |
|-------------------------------|------------|----------|----------------------|----------------------|
| Baseline (FP32)               | 44.90 MB   | 95.77%   | 11.72 ± 0.98 ms      | 250.55 ± 10.66 ms    |
| Half-Precision (FP16)         | 22.52 MB   | 95.77%   | 5.71 ± 0.78 ms       | 5345.78 ± 140.39 ms  |
| Post Training Quant. (INT8)   | 11.39 MB   | 95.62%   | 11.57 ± 0.98 ms      | 145.34 ± 7.18 ms     |
| Quant. Aware Training (INT8)  | 11.39 MB   | 93.08%   | 11.66 ± 0.69 ms      | 141.42 ± 7.83 ms     |


A few things are worth noting from these actual numbers:

- **EfficientNet-B0's INT8 rows are a red flag, not a config typo.** The accuracy of EfficientNet-B0 goes down from 95.29% (FP32) to 69.13% (PTQ) and 23.27% (QAT). This means the FX quantization pipeline is not working well with EfficientNet-B0's architecture. Do not use EfficientNet-B0's quantized weights from this run for anything. This needs to be investigated.
- **The FP16 CPU latency (5345.78 ms) is real and expected, not a bug.** Consumer CPUs do not have native FP16 arithmetic units like GPUs do, so `torch.compile` has to emulate FP16 math on CPU, which is much slower than the FP32 baseline. The FP16 GPU latency (5.71 ms) is where the real speedup shows up. Do not think the CPU FP16 row is an error.
- ResNet18, DenseNet121, GoogLeNet, and MobileNetV2 all work well under INT8 quantization (accuracy loss is less than 2 points). This is what we expect from these models.

### B.6 Known Issues & Fixes (Phase 2)

#### B.6.1 `torch.compile` requires a C/C++ compiler

Both scripts use `torch.compile`, which generates and compiles code on the fly using TorchInductor. **This is not a theoretical risk** — it's the reason the author had to move off native Windows and onto Ubuntu (WSL2) to get these scripts running at all.

- **Linux**: install build tools, e.g. `sudo apt install build-essential` (Debian/Ubuntu/WSL).
- **Windows (not WSL)**: install the *Microsoft C++ Build Tools* (Visual Studio Build Tools, "Desktop development with C++" workload).
- **macOS**: Xcode Command Line Tools (`xcode-select --install`).

If you do not want to deal with this you can remove the `torch.compile(...)` calls and call the model directly. You will lose the compiled-latency numbers. Everything else (size, accuracy) still works.

#### B.6.2 FX quantization backend is x86-only (`fbgemm`)

Both scripts hardcode `get_default_qconfig("fbgemm")` and `get_default_qat_qconfig("fbgemm")`. `fbgemm` is the x86/AMD64 CPU quantization backend. **On ARM CPUs (e.g. Apple Silicon Macs, ARM-based Linux/Windows machines) this will fail or silently produce a non-functional quantized model.** Swap `"fbgemm"` for `"qnnpack"` in both places if you're on ARM.

#### B.6.3 GPU quantization silently does nothing without CUDA

The GPU stages are skipped entirely on a CPU-only machine. This is not a bug. The results table will show `N/A` in the GPU Latency column for every row on such a machine. Do not think the quantization pipeline is broken; it just means no CUDA GPU was detected.

#### B.6.4 `calibration_dataset = Subset(train_dataset, range(512))`

Both scripts hardcode 512 calibration samples for PTQ/QAT calibration. If your `train_dataset` has fewer than 512 images total, this raises an `IndexError`. Reduce `range(512)` to `range(min(512, len(train_dataset)))` if you're working with a smaller dataset.

#### B.6.5 `drop_last=True` on both loaders

`calibration_loader` and `test_loader` both use `drop_last=True`. This means the **last partial batch of your validation set is silently dropped** from both calibration and the reported accuracy. The accuracy numbers are computed over a slightly smaller validation set than Phase 1's `evaluate` step used.

#### B.6.6 `all_models.py` suppresses only specific warnings; `resnet.py` suppresses *all* warnings

`all_models.py` filters three specific known-noisy warning messages. The Phase 2 `resnet.py` has a blanket `warnings.filterwarnings("ignore")` at the top. This silences **all** Python warnings. If something looks wrong with your results and you can't tell why, comment out that line and re-run to see what warnings were being hidden.

#### B.6.7 GoogLeNet aux logits are explicitly disabled here (this is correct, and different from Phase 1)

`get_model_standard` in `all_models.py` sets `model.aux_logits = False`. This is different from the Phase 1 `googlenet.py` approach. This is correct as written.

#### B.6.8 `torchao` / PyTorch version compatibility

`torchao`'s quantization API and FX graph-mode quantization evolve quickly between PyTorch releases. If you hit an `ImportError` or `AttributeError` inside `torchao` or `torch.ao.quantization`, check that your installed `torch` and `torchao` versions were released around the same time.

#### B.6.9 The 16-bit stage in `quantization/resnet.py` is labeled incorrectly

The code changes the model and inputs to `torch.bfloat16`. The results table it prints calls this row "16-Bit (BFloat16)" in the source. However, the actual reference run in Section B.5.1 shows the printed label as `Half-Precision (FP16)`. BFloat16 and FP16 are not the same thing — they split the exponent and mantissa differently and have different numeric ranges and precision behavior. This is a mistake in the label, not a problem with the underlying math. It means you cannot trust the row label alone to know which 16-bit format was actually used. If it matters for your report, check the `.to(...)` call in the code directly, rather than trusting the printed table. Consider fixing the print string to match before you rely on it.

## 7. Disclosure of AI Tool Usage

We disclose the use of AI tools as required by the assignment. **Claude (Anthropic) and Gemini (Google) were used as coding assistants** in Phase 1 and Phase 2. We used them to draft and adapt the training loop boilerplate for the six architectures. We also used them to draft and debug the quantization benchmarking scripts and to debug the issues listed in Section 6 and Section B.6. Additionally, we used them to write this README. All dataset handling, model choices, hyperparameters, and final code were reviewed and run by the author. The AI tools were only used as a coding aid, not as an autonomous agent.
