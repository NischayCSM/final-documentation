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
.
├── models/
│   ├── resnet.py           # Phase 1: trains ResNet18
│   ├── vgg.py               # Phase 1: trains VGG16
│   ├── densenet.py          # Phase 1: trains DenseNet121
│   ├── googlenet.py         # Phase 1: trains GoogLeNet
│   ├── mobilenet.py         # Phase 1: trains MobileNetV2
│   └── efficientnet.py      # Phase 1: trains EfficientNet-B0
├── quantization/
│   ├── resnet.py            # Phase 2: quantization benchmark for ResNet18
│   └── all_models.py        # Phase 2: quantization benchmark for all six architectures
├── requirements.txt
└── README.md
```

Note that there are two different `resnet.py` files in this repository. One is for training. The other is for quantization benchmarking.

Running each script creates the following files:

- `<model>_paddy.pth`. Trained model weights
- `confusion_matrix_<model>.png`. Confusion matrix heatmap
- Console output: per-epoch train/val loss and accuracy, final `classification_report`

## 3. Dataset Setup

We use the Kaggle **Paddy Doctor: Paddy Disease Classification** dataset for this project.

1. Get the dataset from Kaggle.
2. Unzip the dataset. You need the `train_images/` folder and `train.csv`. The `train_images/` folder must have one subfolder for each class.
3. Change the dataset paths in each script.

## 4. Environment Setup

We suggest using Python 3.10 or later.

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4.1 Installing PyTorch

Install PyTorch before installing the rest of the requirements.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the rest:

```bash
pip install -r requirements.txt
```

Check that your GPU is visible before training:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 4.2 First-run internet access

The first time each script runs it will download the pre-trained ImageNet weights for the model. Make sure you have an internet connection.

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

Run them one at a time.

## 6. Known Issues & Fixes

### 6.1 `densenet.py`. CUDA check is not working

Fix the CUDA check in `densenet.py`:

```python
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

### B.1 Things You Need To Do Before Running The Scripts

These scripts need the weights from Phase 1 to be in the working folder. The weights are in files like `resnet_paddy.pth`. If a weight file is missing, the script will use a default ImageNet weight instead. The accuracy with the default weight will not be as good as with the one you trained.

### B.2 The Path To The Dataset

You need to tell the script where the dataset is. The path is in the `data_dir` variable. You need to change this to the path where your dataset lives.

### B.3 The Number Of Classes

The script assumes that there are 10 classes in the dataset. If you have a different number of classes you need to change the `num_classes` variable.

### B.4 Things You Need To Install

You need to install `torchao` to run these scripts. You also need to have a C++ compiler installed.

### B.5 Running The Scripts

You can run the scripts using the following commands:

```bash
python quantization/all_models.py
python quantization/resnet.py
```

These scripts will print the results to the console. If you want to save the results you can redirect the output to a file.

### B.5.1 Example Output

Here is an example of what the output might look like:

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

**`quantization/resnet.py`** — ResNet18 pass (includes the half/16-bit precision stage):

```
### ResNet18
| Method                       | Model Size | Accuracy | GPU Latency (ms)    | CPU Latency (ms)    |
|-------------------------------|------------|----------|----------------------|----------------------|
| Baseline (FP32)               | 44.90 MB   | 95.77%   | 11.72 ± 0.98 ms      | 250.55 ± 10.66 ms    |
| Half-Precision (FP16)         | 22.52 MB   | 95.77%   | 5.71 ± 0.78 ms       | 5345.78 ± 140.39 ms  |
| Post Training Quant. (INT8)   | 11.39 MB   | 95.62%   | 11.57 ± 0.98 ms      | 145.34 ± 7.18 ms     |
| Quant. Aware Training (INT8)  | 11.39 MB   | 93.08%   | 11.66 ± 0.69 ms      | 141.42 ± 7.83 ms     |
```

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

The code changes the model and inputs to `torch.bfloat16`. The results table it prints calls this row "16-Bit (BFloat16)" in the source. However, the actual reference run in §B.5.1 shows the printed label as `Half-Precision (FP16)`. BFloat16 and FP16 are not the same thing — they split the exponent and mantissa differently and have different numeric ranges and precision behavior. This is a mistake in the label, not a problem with the underlying math. It means you cannot trust the row label alone to know which 16-bit format was actually used. If it matters for your report, check the `.to(...)` call in the code directly, rather than trusting the printed table. Consider fixing the print string to match before you rely on it.

## 7. Disclosure of AI Tool Usage

We disclose the use of AI tools as required by the assignment. **Claude (Anthropic) and Gemini (Google) were used as coding assistants** in Phase 1 and Phase 2. We used them to draft and adapt the training loop boilerplate for the six architectures. We also used them to draft and debug the quantization benchmarking scripts and to debug the issues listed in §6 and §B.6. Additionally, we used them to write this README. All dataset handling, model choices, hyperparameters, and final code were reviewed and run by the author. The AI tools were only used as a coding aid, not as an autonomous agent.

## 8. Roadmap for Phase 3 and beyond

In Phase 1 we established baseline accuracy for the six architectures, without hyperparameter search, a test-set inference/submission pipeline, or ensembling. In Phase 2 we added benchmarking for inference efficiency — size, latency, and accuracy trade-offs for the trained models under FP32, BF16, INT8 PTQ, and INT8 QAT. We have not picked a deployment target yet. For later phases we plan to:

- Choose a model and precision combination based on the Phase 2 accuracy, latency, and size trade-off table.
- Tune hyperparameters for the selected model, including learning rate, batch size, and augmentation strength.
- Build an inference pipeline for the test set using the `test_images/` folder from the Kaggle competition, and create a submission file by running it through the chosen quantized model.
- Possibly ensemble the top-performing models, and/or convert the selected quantized model into a deployment format such as ONNX, TorchScript, or a mobile runtime, if on-device inference is a goal.