# Paddy Disease Classification by PyTorch — Baseline Training & Quantization Benchmarking

## Part A — Phase 1: Baseline CNN Training

## 1. Overview

Six independent scripts, one per architecture, each:

1. Loads the paddy disease images with `torchvision.datasets.ImageFolder`.
2. Splits the data into a stratified 80/20 train/validation split.
3. Fine-tunes an ImageNet-pretrained backbone with a replaced classification head.
4. Trains for 10 epochs with Adam + cosine annealing LR scheduling.
5. Saves the trained weights (`.pth`) and a classification report + confusion matrix (`.png`).

| Script | Architecture | Backbone weights |
|---|---|---|
| `resnet.py` | ResNet18 | `ResNet18_Weights.DEFAULT` |
| `vgg.py` | VGG16 | `VGG16_Weights.DEFAULT` |
| `densenet.py` | DenseNet121 | `DenseNet121_Weights.DEFAULT` |
| `googlenet.py` | GoogLeNet (Inception v1) | `GoogLeNet_Weights.DEFAULT` |
| `mobilenet.py` | MobileNetV2 | `MobileNet_V2_Weights.DEFAULT` |
| `efficientnet.py` | EfficientNet-B0 | `EfficientNet_B0_Weights.DEFAULT` |

All six scripts follow the same overall pipeline; they differ only in the backbone, the classifier-head replacement, and (for GoogLeNet) handling of auxiliary classifier outputs.

## 2. Repository Structure

```
.
├── models/
│   ├── resnet.py           # Phase 1: trains ResNet18
│   ├── vgg.py               # Phase 1: trains VGG16
│   ├── densenet.py          # Phase 1: trains DenseNet121
│   ├── googlenet.py         # Phase 1: trains GoogLeNet
│   ├── mobilenet.py         # Phase 1: trains MobileNetV2
│   └── efficientnet.py      # Phase 1: trains EfficientNet-B0
├── quantization/
│   ├── resnet.py            # Phase 2: quantization benchmark for ResNet18 only
│   └── all_models.py        # Phase 2: quantization benchmark for all six architectures
├── requirements.txt
└── README.md
```

Note there are two different `resnet.py` files in this repo — `models/resnet.py` (Phase 1 training) and `quantization/resnet.py` (Phase 2 quantization benchmarking). They live in separate folders precisely to avoid a naming collision; when running commands below, make sure you're pointing at the right one for the phase you're working on.

Running each Phase 1 script produces, in the working directory:
- `<model>_paddy.pth` — trained model weights
- `confusion_matrix_<model>.png` — confusion matrix heatmap
- Console output: per-epoch train/val loss & accuracy, final `classification_report`

## 3. Dataset Setup

This project uses the Kaggle **Paddy Doctor: Paddy Disease Classification** dataset.

1. Download it from Kaggle (competition page linked above), either manually or via the Kaggle CLI:
   ```bash
   pip install kaggle
   kaggle competitions download -c paddy-disease-classification
   ```
2. Unzip it. You need the `train_images/` folder and `train.csv`. `train_images/` must already contain one **subfolder per class** (this is how it ships from Kaggle), e.g.:
   ```
   train_images/
   ├── bacterial_leaf_blight/
   ├── bacterial_leaf_streak/
   ├── blast/
   ├── brown_spot/
   ├── dead_heart/
   ├── downy_mildew/
   ├── hispa/
   ├── normal/
   └── tungro/
   ```
   `ImageFolder` derives class labels directly from these subfolder names — `train.csv` is only read/printed for a quick sanity check in these scripts, it is not otherwise used.

3. **You must update the dataset paths before running.** Every script currently hardcodes the original author's local Windows path:
   ```python
   data_dir = r"C:\Users\nisch\OneDrive\Documents\paddy-disease-classification\train_images"
   csv_path = r"C:\Users\nisch\OneDrive\Documents\paddy-disease-classification\train.csv"
   ```
   Change both lines in **each of the six scripts** to point at wherever you extracted the dataset, e.g. on Linux/Mac:
   ```python
   data_dir = "/home/you/data/paddy-disease-classification/train_images"
   csv_path = "/home/you/data/paddy-disease-classification/train.csv"
   ```
   This is the single most common reason the scripts will fail to run out of the box — `FileNotFoundError` on the dataset path.

## 4. Environment Setup

Python 3.10+ recommended.

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4.1 Installing PyTorch (do this separately, before `requirements.txt`)

`torch`/`torchvision` are deliberately **not** pinned in `requirements.txt` because the correct build depends on your hardware. Install these first with the command from the [official PyTorch install selector](https://pytorch.org/get-started/locally/):

- **GPU (CUDA) machine**, e.g. CUDA 12.1:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- **CPU only**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
Then install the rest:
```bash
pip install -r requirements.txt
```

Verify GPU is actually visible before training (important, see §6.1):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 4.2 First-run internet access

The first time each script runs, torchvision will **download the pretrained ImageNet weights** for that backbone (tens to hundreds of MB per model). Make sure you have an internet connection the first time; after that, weights are cached locally (`~/.cache/torch/hub/checkpoints/` on Linux/Mac, `%USERPROFILE%\.cache\torch\hub\checkpoints\` on Windows) and no further downloads happen.

## 5. Running the Scripts

Each script is standalone and self-contained:

```bash
python models/resnet.py
python models/vgg.py
python models/densenet.py
python models/googlenet.py
python models/mobilenet.py
python models/efficientnet.py
```

Run them one at a time — training all six sequentially on a single GPU/CPU took the original author noticeably long, so budget time accordingly (VGG16 and EfficientNet in particular are heavier). There is no shared driver script in Phase 1; each is run manually.

## 6. Known Issues & Fixes (read before running)

These are the actual issues encountered while developing this phase. Fix them before you run the code, or you will hit the same errors.

### 6.1 `densenet.py` — CUDA check is broken

Line 16 of `densenet.py` is:
```python
device=torch.device("cuda" if torch.cuda.is_available else "cpu")
```
This is missing the `()` after `is_available`. `torch.cuda.is_available` (no parentheses) refers to the *function itself*, which is always truthy — so this line **always** selects `"cuda"`, even on a machine with no GPU / no CUDA build of PyTorch installed. On a CPU-only machine this crashes as soon as a tensor or model is moved with `.to(device)` (`AssertionError: Torch not compiled with CUDA enabled`, or similar).

**Fix** — change it to match the other five scripts:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 6.2 Hardcoded Windows paths

As noted in §3, all six scripts hardcode `data_dir` / `csv_path` to the original author's machine. On any other machine (or a different OS), this throws `FileNotFoundError`. Update the two path lines in every script.

### 6.3 Out-of-memory on GPU (VGG16 / EfficientNet especially)

`batch_size=32` at 224×224 resolution can exceed GPU memory on smaller GPUs (≤6GB), especially for VGG16. If you hit `CUDA out of memory`:
- Lower `batch_size` (e.g. 16 or 8) in the `DataLoader` calls, or
- Close other GPU processes, or
- Run on CPU (slow, but works — see §4.1).

### 6.4 `num_workers=0` in all `DataLoader`s

This is set to `0` (no multiprocessing workers) in every script, which is a safe default for Windows (Windows multiprocessing + Jupyter/spawn issues cause silent hangs or `RuntimeError` otherwise). If you're on Linux/Mac and want faster data loading, you can raise this (e.g. `num_workers=4`), but if you're on Windows, leave it at `0` unless you wrap the `main()` call in `if __name__ == "__main__":` (already done in these scripts) — even then, Windows multiprocessing with `ImageFolder` can be flaky.

### 6.5 GoogLeNet auxiliary classifiers

`googlenet.py` correctly checks:
```python
if isinstance(outputs, models.GoogLeNetOutputs):
    ...
```
This is necessary because `models.googlenet(weights=...)` returns a `GoogLeNetOutputs` namedtuple (main logits + 2 auxiliary logits) in `model.train()` mode, but a plain tensor in `model.eval()` mode. If you refactor this script, keep that branch — removing it will crash training with an `AttributeError` or a shape mismatch in the loss function.

### 6.6 Confusion matrix / report titles are copy-pasted

The `generate_report_and_matrix` helper in `mobilenet.py`, `vgg.py`, `resnet.py`, and `efficientnet.py` all print the header `CLASSIFICATION REPORT (MobileNetV2)` regardless of which model actually ran (a copy-paste leftover). This is cosmetic only — it doesn't affect the actual numbers, but don't be confused by the label when reading console output for, say, `vgg.py`.

### 6.7 `train.csv` is loaded but unused

All scripts `pd.read_csv(csv_path)` and print `.shape` / `.label.value_counts()` purely as a sanity check that the dataset was extracted correctly. Class labels for training actually come from the `train_images/` subfolder names via `ImageFolder`, not from this CSV. If your CSV path is wrong but your image folder path is right, training will still proceed (only the CSV print step fails) — the reverse (right CSV, wrong image folder) will crash immediately.

---

## Part B — Phase 2: Quantization Benchmarking

Phase 2 does **not** retrain anything. It loads the `.pth` weights produced in Phase 1 and benchmarks each model's **model size, inference latency (CPU & GPU), and validation accuracy** across four representations:

| Stage | What it does |
|---|---|
| **Baseline (FP32)** | The Phase 1 model as-is, no quantization, just JIT-traced and `torch.compile`'d for a fair latency comparison. |
| **BFloat16** *(ResNet18 only, `quantization/resnet.py`)* | Casts weights and activations to `bfloat16` — a lightweight, lossy-but-usually-safe precision reduction, no calibration data needed. **Note:** the script's own printed results label this row `Half-Precision (FP16)` even though the code casts to `bfloat16`, not `float16` — a labeling mismatch, not a different actual stage. See §B.6.9. |
| **INT8 Post-Training Quantization (PTQ)** | Quantizes an already-trained FP32 model to INT8 using FX Graph Mode Quantization (`prepare_fx` / `convert_fx`), calibrated on a small slice of the training set (no gradient updates). |
| **INT8 Quantization-Aware Training (QAT)** | Same INT8 target, but simulates quantization *during* a short extra fine-tuning pass (`prepare_qat_fx`), so the model adapts to quantization noise before conversion — this generally recovers more accuracy than PTQ, at the cost of extra compute. |

Two scripts are provided:

- **`quantization/all_models.py`** — runs Baseline / PTQ / QAT (no BFloat16 stage) for **all six** Phase 1 architectures in one loop, and prints one results table per model at the end.
- **`quantization/resnet.py`** — the deeper, single-architecture version. ResNet18 was selected as the strongest/most practical candidate from Phase 1, so this script adds the extra **BFloat16** stage on top of Baseline / PTQ / QAT for that one architecture.

### B.1 Prerequisites — this is not a standalone script

Unlike Phase 1, these scripts **expect the Phase 1 `.pth` files to already exist in the working directory**:

```
resnet_paddy.pth
densenet_paddy.pth
efficientnet_paddy.pth
mobilenet_paddy.pth
googlenet_paddy.pth
vgg_paddy.pth
```

If a given `.pth` file is missing, the script does **not** fail — it prints a `⚠️ WARNING ... Falling back to default ImageNet weights` and proceeds anyway, silently benchmarking an **un-fine-tuned ImageNet model** on your paddy validation set. The size/latency numbers will still be meaningful, but the accuracy numbers will be close to random and **not representative of your actual trained models**. Always check the console for these warnings before trusting the accuracy column in the results table. Run the corresponding Phase 1 script first if a warning appears.

### B.2 Dataset path (again — different variable style this time)

```python
data_dir = r"/mnt/c/Datasets/paddy-disease-classification/train_images"
```

This is a **WSL-style path** (`/mnt/c/...`), different from the Windows-native paths used in the Phase 1 scripts. Update it to wherever your dataset actually lives:
- WSL/Linux/Mac: `data_dir = "/home/you/data/paddy-disease-classification/train_images"`
- Native Windows Python (not WSL): `data_dir = r"C:\Users\you\data\paddy-disease-classification\train_images"`

Unlike Phase 1, these scripts check `os.path.exists(data_dir)` up front and print a `WARNING: Dataset path not found.` instead of crashing immediately — but if the path is wrong, `test_loader`/`calibration_loader` will never get defined, and the script **will still crash later** with a `NameError` once it reaches the benchmarking loop. Don't be misled by the graceful-looking warning — fix the path regardless.

### B.3 `num_classes=10` is hardcoded

Both scripts hardcode `num_classes=10` in the model factory. This **must match** the number of classes your Phase 1 `ImageFolder` found (i.e. the number of subfolders in `train_images/`, which is 10 for the full Paddy Doctor dataset). If you trained Phase 1 on a subset of classes, update `num_classes` here to match — otherwise `load_state_dict` will fail with a size-mismatch error on the final classifier layer.

### B.4 Additional dependencies

Phase 2 needs everything from Phase 1, **plus**:
- **`torchao`** — provides `quantize_` and `int8_dynamic_activation_int8_weight` for the GPU-side quantization path.
- A **working C++ compiler on your system PATH** — `torch.compile` (used for every stage, including baseline) with `torch._inductor.config.cpp_wrapper = True` compiles a native C++/CUDA extension under the hood. Without a compiler this fails, typically with a `CalledProcessError` or `distutils`/`cl.exe`/`g++ not found` error. See §B.6 below.

Update `requirements.txt` per §4 above, then also run:
```bash
pip install torchao
```
(already included in the updated `requirements.txt` for this repo — see below).

### B.5 Running the scripts

```bash
python quantization/all_models.py     # all six architectures, Baseline + PTQ + QAT
python quantization/resnet.py          # ResNet18 only, Baseline + BF16 + PTQ + QAT (deeper pass)
```

Both are single self-contained scripts — no CLI arguments. Expect this to run considerably longer than a single Phase 1 script: each model is loaded, benchmarked, quantized, and re-benchmarked up to 4 times, and `torch.compile` itself adds warm-up compilation time on first call for every stage.

> **Platform note:** both scripts were actually run on **native Windows and failed** — `torch.compile`'s use of TorchInductor (`config.cpp_wrapper = True`) needs a working C/C++ toolchain, which a base Windows Python install doesn't have (see §B.6.1). The author switched to **Ubuntu (WSL2)** to get a working C/C++ toolchain via `build-essential`, and both scripts ran successfully there. If you're on Windows, either install the MSVC Build Tools first or just use WSL2/a native Linux machine from the start — it's the more reliable path for this phase.

Output is a Markdown-formatted results table per model printed to the console. These tables are printed only — they are not saved to a file. If you want to keep the results, redirect console output when running, e.g.:
```bash
python quantization/resnet.py | tee resnet_quant_results.txt
```

### B.5.1 Actual reference output (Ubuntu, one real run)

These are real results from one run of both scripts on Ubuntu (WSL2), included so you have something to sanity-check your own run against — expect your own numbers to vary somewhat depending on hardware, PyTorch/torchao versions, and how well your `.pth` weights were trained in Phase 1.

**`quantization/all_models.py`** — all six architectures:

```
### ResNet18
| Method                         | Model Size   | Accuracy   | GPU Latency (ms)     | CPU Latency (ms)     |
|---------------------------------|---------------|-------------|------------------------|------------------------|
| Baseline (no quantization)      | 44.90 MB      | 95.77%      | 12.15 ± 1.15 ms         | 253.65 ± 13.13 ms       |
| Post Training Quantization      | 11.39 MB      | 95.82%      | 11.75 ± 1.03 ms         | 147.39 ± 9.28 ms        |
| Quantization Aware Training     | 11.39 MB      | 94.42%      | 11.68 ± 1.04 ms         | 142.71 ± 7.92 ms        |

### DenseNet121
| Method                         | Model Size   | Accuracy   | GPU Latency (ms)     | CPU Latency (ms)     |
|---------------------------------|---------------|-------------|------------------------|------------------------|
| Baseline (no quantization)      | 29.08 MB      | 96.06%      | 41.04 ± 0.40 ms         | 971.96 ± 51.60 ms       |
| Post Training Quantization      | 8.47 MB       | 95.05%      | 41.37 ± 0.42 ms         | 281.14 ± 10.96 ms       |
| Quantization Aware Training     | 8.47 MB       | 93.99%      | 42.15 ± 1.50 ms         | 298.35 ± 19.99 ms       |

### EfficientNet_B0
| Method                         | Model Size   | Accuracy   | GPU Latency (ms)     | CPU Latency (ms)     |
|---------------------------------|---------------|-------------|------------------------|------------------------|
| Baseline (no quantization)      | 16.83 MB      | 95.29%      | 15.26 ± 0.57 ms         | 151.25 ± 10.00 ms       |
| Post Training Quantization      | 5.24 MB       | 69.13%      | 15.55 ± 0.57 ms         | 176.30 ± 10.16 ms       |
| Quantization Aware Training     | 5.24 MB       | 23.27%      | 15.34 ± 0.37 ms         | 179.09 ± 4.13 ms        |

### MobileNet_V2
| Method                         | Model Size   | Accuracy   | GPU Latency (ms)     | CPU Latency (ms)     |
|---------------------------------|---------------|-------------|------------------------|------------------------|
| Baseline (no quantization)      | 9.48 MB       | 92.64%      | 13.19 ± 0.55 ms         | 264.16 ± 14.31 ms       |
| Post Training Quantization      | 2.92 MB       | 89.76%      | 13.37 ± 0.51 ms         | 64.05 ± 2.78 ms         |
| Quantization Aware Training     | 2.92 MB       | 80.96%      | 13.18 ± 0.42 ms         | 62.61 ± 3.40 ms         |

### GoogLeNet
| Method                         | Model Size   | Accuracy   | GPU Latency (ms)     | CPU Latency (ms)     |
|---------------------------------|---------------|-------------|------------------------|------------------------|
| Baseline (no quantization)      | 22.94 MB      | 93.51%      | 13.86 ± 0.68 ms         | 236.39 ± 8.22 ms        |
| Post Training Quantization      | 6.13 MB       | 93.22%      | 13.93 ± 0.80 ms         | 137.25 ± 7.66 ms        |
| Quantization Aware Training     | 6.13 MB       | 92.45%      | 13.95 ± 0.91 ms         | 139.56 ± 6.95 ms        |

### VGG16
| Method                         | Model Size   | Accuracy   | GPU Latency (ms)     | CPU Latency (ms)     |
|---------------------------------|---------------|-------------|------------------------|------------------------|
| Baseline (no quantization)      | 537.28 MB     | 94.66%      | 81.04 ± 0.41 ms         | 1619.35 ± 32.14 ms      |
| Post Training Quantization      | 134.63 MB     | 94.90%      | 80.79 ± 0.39 ms         | 849.39 ± 10.41 ms       |
| Quantization Aware Training     | 134.63 MB     | 90.62%      | 80.74 ± 0.33 ms         | 847.85 ± 12.95 ms       |
```

**`quantization/resnet.py`** — ResNet18 deep pass (includes the half/16-bit precision stage):

```
### ResNet18
| Method                         | Model Size   | Accuracy   | GPU Latency (ms)     | CPU Latency (ms)     |
|---------------------------------|---------------|-------------|------------------------|------------------------|
| Baseline (FP32)                 | 44.90 MB      | 95.77%      | 11.72 ± 0.98 ms         | 250.55 ± 10.66 ms       |
| Half-Precision (FP16)           | 22.52 MB      | 95.77%      | 5.71 ± 0.78 ms          | 5345.78 ± 140.39 ms     |
| Post Training Quant. (INT8)     | 11.39 MB      | 95.62%      | 11.57 ± 0.98 ms         | 145.34 ± 7.18 ms        |
| Quant. Aware Training (INT8)    | 11.39 MB      | 93.08%      | 11.66 ± 0.69 ms         | 141.42 ± 7.83 ms        |
```

A few things worth noting from these actual numbers:
- **EfficientNet-B0's INT8 rows are a red flag, not a config typo.** Accuracy collapses from 95.29% (FP32) to 69.13% (PTQ) and 23.27% (QAT) — QAT accuracy *dropping below* PTQ is a strong sign the FX quantization pipeline is quantizing something in EfficientNet's architecture (its `SiLU`/Swish activations and squeeze-excitation blocks are known to be quantization-sensitive) badly, not that quantization is simply "lossy" for this model. Don't use EfficientNet's quantized weights from this run for anything downstream — treat this as a Phase 3 investigation item, not an accepted result.
- **The FP16 CPU latency (5345.78 ms) is real, and expected, not a bug.** Consumer CPUs don't have native FP16 arithmetic units the way GPUs do — `torch.compile` has to emulate FP16 math on CPU, which is dramatically slower than the FP32 baseline. FP16 GPU latency (5.71 ms) is where the real speedup shows up. Read the CPU FP16 row as "don't deploy FP16 on CPU," not as an error.
- ResNet18, DenseNet121, GoogLeNet, and MobileNetV2 all hold up well under INT8 quantization (accuracy loss well under 2 points), which is consistent with these being the more quantization-friendly, "reference-standard" convolutional backbones vs. VGG16's very large fully-connected classifier head or EfficientNet's newer/mobile-optimized ops.

### B.6 Known Issues & Fixes (Phase 2)

#### B.6.1 `torch.compile` requires a C/C++ compiler

Both scripts set `torch._inductor.config.cpp_wrapper = True` and wrap **every** stage — even the FP32 baseline — in `torch.compile(...)`. This uses TorchInductor, which generates and compiles native code on the fly. **This is not a theoretical risk** — it's the actual reason the author had to move off native Windows and onto Ubuntu (WSL2) to get these scripts running at all (see the platform note in §B.5).
- **Linux**: install build tools first, e.g. `sudo apt install build-essential` (Debian/Ubuntu/WSL).
- **Windows (native, not WSL)**: install the *Microsoft C++ Build Tools* (Visual Studio Build Tools, "Desktop development with C++" workload) — plain `cl.exe`-less Windows Python installs will fail here. Using WSL avoids this entirely.
- **macOS**: Xcode Command Line Tools (`xcode-select --install`).

If you don't want to deal with this, you can remove `torch.compile(...)` calls and call the model directly — you'll lose the compiled-latency numbers but everything else (size, accuracy) still works. This is the fastest workaround if you just want the quantization accuracy numbers and don't care about compiled latency.

#### B.6.2 FX quantization backend is x86-only (`fbgemm`)

Both scripts hardcode:
```python
get_default_qconfig("fbgemm")
get_default_qat_qconfig("fbgemm")
```
`fbgemm` is the x86/AMD64 CPU quantization backend. **On ARM CPUs (e.g. Apple Silicon Macs, ARM-based Linux/Windows machines), this will fail or silently produce a non-functional quantized model.** Swap `"fbgemm"` for `"qnnpack"` in both places if you're on ARM:
```python
qconfig_ptq = get_default_qconfig("qnnpack")
qconfig_qat = get_default_qat_qconfig("qnnpack")
```

#### B.6.3 GPU quantization silently does nothing without CUDA

The GPU stages (`if torch.cuda.is_available(): ...`) are skipped entirely on a CPU-only machine — this is intentional, not a bug, but the results table will show `N/A` in the GPU Latency column for every row on such a machine. Don't mistake that for a broken quantization pipeline; it just means no CUDA GPU was detected.

#### B.6.4 `calibration_dataset = Subset(train_dataset, range(512))`

Both scripts hardcode 512 calibration samples for PTQ/QAT calibration. If your `train_dataset` (80% split) has fewer than 512 images total — e.g. you're testing on a small subset of the dataset rather than the full Paddy Doctor set — this raises an `IndexError`. Reduce `range(512)` to `range(min(512, len(train_dataset)))` if you're working with a smaller dataset.

#### B.6.5 `drop_last=True` on both loaders

`calibration_loader` and `test_loader` both use `drop_last=True`, and `estimate_latency`/quantization tracing use a fixed `batch_size=32` example input. This means the **last partial batch of your validation set is silently dropped** from both calibration and the reported accuracy — the accuracy numbers in Phase 2 are computed over a slightly smaller validation set than Phase 1's `evaluate` step used. Not a bug that will crash anything, but worth knowing if you're comparing Phase 1 vs. Phase 2 accuracy numbers directly and they don't match exactly.

#### B.6.6 `all_models.py` suppresses only specific warnings; `resnet.py` suppresses *all* warnings

`all_models.py` filters three specific known-noisy warning messages. The Phase 2 `resnet.py`, by contrast, has a blanket `warnings.filterwarnings("ignore")` at the top — this silences **all** Python warnings, including ones that might indicate a real problem (e.g. a silent precision downcast, a deprecated API about to break in a future PyTorch version). If something looks wrong with your results and you can't tell why, temporarily comment out that line and re-run to see what warnings were being hidden.

#### B.6.7 GoogLeNet aux logits are explicitly disabled here (this is correct, and different from Phase 1)

`get_model_standard` in `all_models.py` sets `model.aux_logits = False` right after construction. This is different from — and simpler than — the Phase 1 `googlenet.py` approach of checking `isinstance(outputs, models.GoogLeNetOutputs)` at each forward pass, because Phase 2 only ever runs the model in `.eval()` for inference/quantization, never trains it from scratch with the auxiliary heads. This is correct as written; just don't be confused if you compare the two scripts side by side and see them handle GoogLeNet differently.

#### B.6.8 `torchao` / PyTorch version compatibility

`torchao`'s quantization API and FX graph-mode quantization (`torch.ao.quantization.quantize_fx`) both evolve quickly between PyTorch releases, and older/newer combinations can mismatch (missing functions, changed signatures). If you hit an `ImportError` or `AttributeError` inside `torchao` or `torch.ao.quantization`, check that your installed `torch` and `torchao` versions were released around the same time — see the `torchao` [compatibility notes](https://github.com/pytorch/ao) for the currently supported PyTorch version range, and reinstall matching versions if needed.

#### B.6.9 `quantization/resnet.py`'s 16-bit stage is mislabeled

The code casts the model and inputs with `.to(torch.bfloat16)`, but the results table it prints calls this row `"16-Bit (BFloat16)"` in the source, yet the actual reference run in §B.5.1 shows the printed label as `Half-Precision (FP16)`. BFloat16 and FP16 are **not the same format** (different exponent/mantissa split, different numeric range and precision behavior) — this is a cosmetic string mismatch like the copy-pasted report titles in Phase 1 (§6.6), not a sign the wrong math ran, but it means you can't trust the row label alone to know which 16-bit format was actually used. If it matters for your write-up, check the `.to(...)` call in the code directly rather than the printed table, and consider fixing the print string to match before you rely on it for a report.

## 7. AI Tool Usage Disclosure

In line with the assignment requirement to declare AI tool usage: **Claude (Anthropic) and Gemini (Google) were used as coding assistants** throughout Phase 1 and Phase 2 — for drafting/adapting the training loop boilerplate across the six architectures, drafting/debugging the quantization benchmarking scripts, debugging the issues listed in §6 and §B.6, and for writing this README. All dataset handling, model choices, hyperparameters, and final code were reviewed and run by the author; the AI tools were used as a coding aid, not as an autonomous agent.

## 8. Roadmap (Phase 3 and beyond)

Phase 1 established baseline accuracy across six architectures with no hyperparameter search, no test-set inference/submission pipeline, and no ensembling. Phase 2 added inference-efficiency benchmarking (size/latency/accuracy trade-offs under FP32, BF16, INT8 PTQ, and INT8 QAT) for those trained models, without yet picking a final deployment target. Planned for later phases:
- Formally select a final model + precision combination based on the Phase 2 accuracy/latency/size trade-off table (not just raw Phase 1 validation accuracy).
- Hyperparameter tuning (learning rate, batch size, augmentation strength) on the chosen architecture(s).
- Proper test-set inference pipeline (`test_images/` from the Kaggle competition) with a submission file, run through the selected quantized model.
- Possible ensembling of the top-performing models, and/or exporting the chosen quantized model to a deployment format (e.g. ONNX, TorchScript, mobile runtime) if on-device inference is a goal.

## 9. License

Add your preferred license here (e.g. MIT) if you intend to make this repository public.
