import os
import sys

# Automatically configure WSL NVIDIA driver library paths for torch.compile
if "LIBRARY_PATH" not in os.environ or "/usr/lib/wsl/lib" not in os.environ["LIBRARY_PATH"]:
    os.environ["LIBRARY_PATH"] = f"/usr/lib/wsl/lib:{os.environ.get('LIBRARY_PATH', '')}"

if "LD_LIBRARY_PATH" not in os.environ or "/usr/lib/wsl/lib" not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"/usr/lib/wsl/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import argparse
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx
from torch.ao.quantization.qconfig import get_default_qat_qconfig, get_default_qconfig

import torchao
from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
import torch._inductor.config as config

warnings.filterwarnings("ignore")
config.cpp_wrapper = True

def parse_args():
    parser = argparse.ArgumentParser(description="Universal Precision & Quantization Benchmark")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training images dataset")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing the trained .pth weight files")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["resnet18", "densenet121", "efficientnet_b0", "mobilenet_v2", "vgg16", "googlenet", "all"],
                        help="Select the model to benchmark, or 'all' to run the full suite")
    return parser.parse_args()

def build_model(model_name, num_classes=10):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "googlenet":
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    for m in model.modules():
        if hasattr(m, 'inplace'):
            m.inplace = False
    return model

def print_size_of_model(model, tag="", example_input=None):
    temp_filename = "temp_serialized_model.pt"
    try:
        if example_input is not None:
            model.eval()
            traced = torch.jit.trace(model, example_input, strict=False)
            torch.jit.save(traced, temp_filename)
        else:
            torch.save(model.state_dict(), temp_filename)
        size_mb_full = os.path.getsize(temp_filename) / 1e6
    except Exception:
        torch.save(model.state_dict(), temp_filename)
        size_mb_full = os.path.getsize(temp_filename) / 1e6
        
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return size_mb_full

def evaluate_accuracy(model, loader, device="cpu", dtype=torch.float32):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, dtype=dtype)
            if device == "cpu":
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

class Timer:
    def __init__(self, device_type):
        self.device_type = device_type
        if self.device_type == "cuda":
            self.starter = torch.cuda.Event(enable_timing=True)
            self.ender = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.device_type == "cuda":
            self.starter.record()
        else:
            self.start_time = time.time()

    def stop(self):
        if self.device_type == "cuda":
            self.ender.record()
            torch.cuda.synchronize()
            return self.starter.elapsed_time(self.ender)  
        else:
            return (time.time() - self.start_time) * 1000  

def estimate_latency(model, example_inputs, device_type, repetitions=50):
    timer = Timer(device_type)
    timings = np.zeros((repetitions, 1))
    for _ in range(5):
        _ = model(example_inputs)
    with torch.no_grad():
        for rep in range(repetitions):
            timer.start()
            _ = model(example_inputs)
            elapsed = timer.stop()
            timings[rep] = elapsed
    return np.mean(timings), np.std(timings)

def main():
    args = parse_args()
    print(f"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset path '{args.data_dir}' not found.")
    if not os.path.exists(args.weights_dir):
        raise FileNotFoundError(f"Weights directory '{args.weights_dir}' not found.")

    # 1. Dataset Setup
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_full = datasets.ImageFolder(root=args.data_dir, transform=val_transforms)
    val_full = datasets.ImageFolder(root=args.data_dir, transform=val_transforms)
    num_classes = len(train_full.classes)

    targets = train_full.targets
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=0.2, random_state=42, stratify=targets
    )

    calibration_dataset = Subset(Subset(train_full, train_idx), range(min(512, len(train_idx))))
    calibration_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(Subset(val_full, val_idx), batch_size=32, shuffle=False, drop_last=True)

    cpu_input_fp32 = torch.rand(32, 3, 224, 224, device="cpu").contiguous(memory_format=torch.channels_last)
    cuda_input_fp32 = torch.rand(32, 3, 224, 224, device="cuda").contiguous(memory_format=torch.channels_last) if torch.cuda.is_available() else None

    # Determine Models to Run
    model_list = ["resnet18", "densenet121", "efficientnet_b0", "mobilenet_v2", "googlenet", "vgg16"] if args.model == "all" else [args.model]
    
    global_results_summary = []

    for model_name in model_list:
        print("\n" + "="*80)
        print(f"🚀 EXECUTING PIPELINE FOR: {model_name.upper()}")
        print("="*80)

        weights_path = os.path.join(args.weights_dir, f"{model_name}_paddy.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights missing! Make sure '{weights_path}' exists.")
        print(f"Loaded custom weights: '{weights_path}'")

        if model_name == "efficientnet_b0":
            print("⚠️ NOTE: EfficientNet undergoes severe accuracy degradation under FX INT8 quantization.")

        # --- BASELINE (FP32) ---
        print("\n--- 1. Benchmarking Baseline (FP32) ---")
        model_fp32 = build_model(model_name, num_classes).eval()
        model_fp32.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

        base_size = print_size_of_model(model_fp32, "Baseline", cpu_input_fp32)
        base_acc = evaluate_accuracy(model_fp32, test_loader, device="cpu")

        torch._dynamo.reset()
        optimized_cpu = torch.compile(model_fp32.to("cpu"))
        with torch.no_grad():
            optimized_cpu(cpu_input_fp32) 
            cpu_mu, cpu_std = estimate_latency(optimized_cpu, cpu_input_fp32, "cpu")

        cuda_mu, cuda_std = 0.0, 0.0
        if torch.cuda.is_available():
            torch._dynamo.reset()
            optimized_cuda = torch.compile(model_fp32.to("cuda"))
            with torch.no_grad():
                optimized_cuda(cuda_input_fp32)
                cuda_mu, cuda_std = estimate_latency(optimized_cuda, cuda_input_fp32, "cuda")

        global_results_summary.append((model_name, "Baseline (FP32)", base_size, cuda_mu, cpu_mu, base_acc, cuda_std, cpu_std))

        # --- BFloat16 ---
        print("\n--- 2. Benchmarking 16-Bit (BFloat16) ---")
        model_bf16 = build_model(model_name, num_classes).eval()
        model_bf16.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        model_bf16 = model_bf16.to(torch.bfloat16)
        
        cpu_input_bf16 = cpu_input_fp32.to(torch.bfloat16)
        cuda_input_bf16 = cuda_input_fp32.to(torch.bfloat16) if torch.cuda.is_available() else None

        bf16_size = print_size_of_model(model_bf16, "BFloat16", cpu_input_bf16)
        eval_device = "cuda" if torch.cuda.is_available() else "cpu"
        bf16_acc = evaluate_accuracy(model_bf16.to(eval_device), test_loader, device=eval_device, dtype=torch.bfloat16)

        bf16_cpu_mu, bf16_cpu_std, bf16_cuda_mu, bf16_cuda_std = 0.0, 0.0, 0.0, 0.0
        
        # PREVENT UBUNTU CRASH: Skip compilation for VGG16
        if model_name == "vgg16":
            print("⚠️ WARNING: Skipping BFloat16 latency compilation for VGG16 to prevent OS-level Out-Of-Memory (OOM) crashes.")
        else:
            try:
                torch._dynamo.reset()
                optimized_cpu_bf16 = torch.compile(model_bf16.to("cpu"))
                with torch.no_grad():
                    optimized_cpu_bf16(cpu_input_bf16)
                    bf16_cpu_mu, bf16_cpu_std = estimate_latency(optimized_cpu_bf16, cpu_input_bf16, "cpu")
            except Exception as e:
                print(f"⚠️ CPU BFloat16 compilation skipped: {e}")

            if torch.cuda.is_available():
                try:
                    torch._dynamo.reset()
                    optimized_cuda_bf16 = torch.compile(model_bf16.to("cuda"))
                    with torch.no_grad():
                        optimized_cuda_bf16(cuda_input_bf16) 
                        bf16_cuda_mu, bf16_cuda_std = estimate_latency(optimized_cuda_bf16, cuda_input_bf16, "cuda")
                except Exception as e:
                    print(f"⚠️ CUDA BFloat16 compilation skipped: {e}")

        global_results_summary.append((model_name, "16-Bit (BFloat16)", bf16_size, bf16_cuda_mu, bf16_cpu_mu, bf16_acc, bf16_cuda_std, bf16_cpu_std))

        # --- INT8 PTQ ---
        print("\n--- 3. Post Training Quantization (INT8) ---")
        cpu_model_ptq = build_model(model_name, num_classes).eval().to("cpu")
        cpu_model_ptq.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        qconfig_ptq = get_default_qconfig("fbgemm")
        cpu_model_ptq.qconfig = qconfig_ptq
        prepared_ptq_fx = prepare_fx(cpu_model_ptq, {"": qconfig_ptq}, example_inputs=cpu_input_fp32)

        with torch.no_grad():
            for image, _ in calibration_loader:
                prepared_ptq_fx(image.to("cpu", memory_format=torch.channels_last))

        quantized_model_cpu = convert_fx(prepared_ptq_fx)
        ptq_size = print_size_of_model(quantized_model_cpu, "PTQ", cpu_input_fp32)
        ptq_acc = evaluate_accuracy(quantized_model_cpu, test_loader, device="cpu")

        torch._dynamo.reset()
        optimized_quant_cpu = torch.compile(quantized_model_cpu)
        with torch.no_grad():
            optimized_quant_cpu(cpu_input_fp32)
            ptq_cpu_mu, ptq_cpu_std = estimate_latency(optimized_quant_cpu, cpu_input_fp32, "cpu")

        ptq_cuda_mu, ptq_cuda_std = 0.0, 0.0
        if torch.cuda.is_available():
            cuda_model_ptq = build_model(model_name, num_classes).eval().to("cuda")
            cuda_model_ptq.load_state_dict(torch.load(weights_path, map_location="cuda", weights_only=True))
            quantize_(cuda_model_ptq, int8_dynamic_activation_int8_weight())
            torch._dynamo.reset()
            optimized_quant_cuda = torch.compile(cuda_model_ptq)
            with torch.no_grad():
                optimized_quant_cuda(cuda_input_fp32) 
                ptq_cuda_mu, ptq_cuda_std = estimate_latency(optimized_quant_cuda, cuda_input_fp32, "cuda")

        global_results_summary.append((model_name, "Post Training Quant (INT8)", ptq_size, ptq_cuda_mu, ptq_cpu_mu, ptq_acc, ptq_cuda_std, ptq_cpu_std))

        # --- INT8 QAT ---
        print("\n--- 4. Quantization Aware Training (INT8) ---")
        cpu_model_qat = build_model(model_name, num_classes).train().to("cpu")
        cpu_model_qat.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        qconfig_qat = get_default_qat_qconfig("fbgemm")
        cpu_model_qat.qconfig = qconfig_qat
        prepared_model_qat = prepare_qat_fx(cpu_model_qat, {"": qconfig_qat}, example_inputs=cpu_input_fp32)

        optimizer = optim.Adam(prepared_model_qat.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        batch_count = 0
        for images, labels in calibration_loader:
            optimizer.zero_grad()
            loss = criterion(prepared_model_qat(images.to("cpu", memory_format=torch.channels_last)), labels.to("cpu"))
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count >= 10: break

        prepared_model_qat.eval()
        quantized_model_qat = convert_fx(prepared_model_qat)
        qat_size = print_size_of_model(quantized_model_qat, "QAT", cpu_input_fp32)
        qat_acc = evaluate_accuracy(quantized_model_qat, test_loader, device="cpu")

        torch._dynamo.reset()
        optimized_qat_cpu = torch.compile(quantized_model_qat)
        with torch.no_grad():
            optimized_qat_cpu(cpu_input_fp32)
            qat_cpu_mu, qat_cpu_std = estimate_latency(optimized_qat_cpu, cpu_input_fp32, "cpu")

        qat_cuda_mu, qat_cuda_std = 0.0, 0.0
        if torch.cuda.is_available():
            cuda_model_qat = build_model(model_name, num_classes).eval().to("cuda")
            cuda_model_qat.load_state_dict(cpu_model_qat.state_dict())
            quantize_(cuda_model_qat, int8_dynamic_activation_int8_weight())
            torch._dynamo.reset()
            optimized_qat_cuda = torch.compile(cuda_model_qat)
            with torch.no_grad():
                optimized_qat_cuda(cuda_input_fp32) 
                qat_cuda_mu, qat_cuda_std = estimate_latency(optimized_qat_cuda, cuda_input_fp32, "cuda")

        global_results_summary.append((model_name, "Quant Aware Train (INT8)", qat_size, qat_cuda_mu, qat_cpu_mu, qat_acc, qat_cuda_std, qat_cpu_std))

    # --- Print Summary Table ---
    print("\n" + "="*105)
    print("FINAL BENCHMARKING RESULTS")
    print("="*105)

    grouped_results = {m: [] for m in model_list}
    for row in global_results_summary:
        grouped_results[row[0]].append(row)

    for m_name in model_list:
        print(f"\n### {m_name.upper()}")
        print(f"| {'Method':<30} | {'Model Size':<12} | {'Accuracy':<10} | {'GPU Latency (ms)':<20} | {'CPU Latency (ms)':<20} |")
        print(f"|{'-'*32}|{'-'*14}|{'-'*12}|{'-'*22}|{'-'*22}|")
        for _, method, size, cuda_l, cpu_l, acc, cuda_s, cpu_s in grouped_results[m_name]:
            cpu_str = f"{cpu_l:.2f} ± {cpu_s:.2f} ms" if cpu_l > 0 else "N/A"
            cuda_str = f"{cuda_l:.2f} ± {cuda_s:.2f} ms" if cuda_l > 0 else "N/A"
            print(f"| {method:<30} | {size:.2f} MB   | {acc:.2f}%     | {cuda_str:<20} | {cpu_str:<20} |")
    print("\n")

if __name__ == "__main__":
    main()