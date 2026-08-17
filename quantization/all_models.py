import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# FX Quantization Imports
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx
from torch.ao.quantization.qconfig import get_default_qat_qconfig, get_default_qconfig

# TorchAO Imports
import torchao
from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
import torch._inductor.config as config

# ==========================================
# CONTEXT & SETUP
# ==========================================
warnings.filterwarnings("ignore", message=".*erase_node(.*) on an already erased node.*")
warnings.filterwarnings("ignore", message=".*The input of maxpool2d is not quantized.*")
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*")
config.cpp_wrapper = True

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = r"/mnt/c/Datasets/paddy-disease-classification/train_images"

if not os.path.exists(data_dir):
    print("WARNING: Dataset path not found.")
else:
    train_full = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    val_full = datasets.ImageFolder(root=data_dir, transform=val_transforms)

    targets = train_full.targets
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=0.2, random_state=42, stratify=targets
    )

    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(val_full, val_idx)

    calibration_dataset = Subset(train_dataset, range(512))
    calibration_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)

# ==========================================
# MODEL FACTORY
# ==========================================
def get_model_standard(model_name, num_classes=10):
    if model_name == "resnet18":
        model = models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights="DEFAULT")
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights="DEFAULT")
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "googlenet":
        model = models.googlenet(weights="DEFAULT")
        model.aux_logits = False 
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    for m in model.modules():
        if hasattr(m, 'inplace'):
            m.inplace = False
    return model

# ==========================================
# METRICS TOOLS
# ==========================================
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
        
    print(f"Size ({tag}): {size_mb_full:.2f} MB")
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return size_mb_full

def evaluate_accuracy(model, loader, device="cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if device == "cpu":
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

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


# ==========================================
# MAIN BENCHMARKING LOOP
# ==========================================
models_to_test = ["resnet18", "densenet121", "efficientnet_b0", "mobilenet_v2", "googlenet", "vgg16"]

file_name_map = {
    "resnet18": "resnet_paddy.pth",
    "densenet121": "densenet_paddy.pth",
    "efficientnet_b0": "efficientnet_paddy.pth",
    "mobilenet_v2": "mobilenet_paddy.pth",
    "googlenet": "googlenet_paddy.pth",
    "vgg16": "vgg_paddy.pth"
}

# Used to format the headers nicely at the end
display_names_map = {
    "resnet18": "ResNet18",
    "densenet121": "DenseNet121",
    "efficientnet_b0": "EfficientNet_B0",
    "mobilenet_v2": "MobileNet_V2",
    "googlenet": "GoogLeNet",
    "vgg16": "VGG16"
}

global_results_summary = []

cpu_input = torch.rand(32, 3, 224, 224, device="cpu").contiguous(memory_format=torch.channels_last)
cuda_input = torch.rand(32, 3, 224, 224, device="cuda").contiguous(memory_format=torch.channels_last) if torch.cuda.is_available() else None

for current_model_name in models_to_test:
    print("\n" + "="*80)
    print(f"🚀 EXECUTING PIPELINE FOR: {display_names_map[current_model_name]}")
    print("="*80)
    
    weights_path = file_name_map[current_model_name]
    
    if os.path.exists(weights_path):
        print(f"✅ Successfully found custom weights: '{weights_path}'")
    else:
        print(f"⚠️ WARNING: '{weights_path}' not found! Falling back to default ImageNet weights.")
        
    # ------------------------------------------
    # 1. BASELINE MODEL BENCHMARKING
    # ------------------------------------------
    print(f"\n--- Benchmarking Baseline (FP32) ---")
    model_to_quantize = get_model_standard(current_model_name, num_classes=10).eval()
    if os.path.exists(weights_path):
        model_to_quantize.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

    base_size = print_size_of_model(model_to_quantize, "Baseline", cpu_input)
    base_acc = evaluate_accuracy(model_to_quantize, test_loader, device="cpu")

    torch._dynamo.reset()
    cpu_model = model_to_quantize.to("cpu")
    with torch.no_grad():
        optimized_cpu = torch.compile(cpu_model)
        optimized_cpu(cpu_input) 
        cpu_mu, cpu_std = estimate_latency(optimized_cpu, cpu_input, "cpu")

    cuda_mu, cuda_std = 0.0, 0.0
    if torch.cuda.is_available():
        torch._dynamo.reset()
        cuda_model = model_to_quantize.to("cuda")
        with torch.no_grad():
            optimized_cuda = torch.compile(cuda_model)
            optimized_cuda(cuda_input)
            cuda_mu, cuda_std = estimate_latency(optimized_cuda, cuda_input, "cuda")

    global_results_summary.append((current_model_name, "Baseline (no quantization)", base_size, cuda_mu, cpu_mu, base_acc, cuda_std, cpu_std))

    # ------------------------------------------
    # 2. POST-TRAINING QUANTIZATION (PTQ)
    # ------------------------------------------
    print("\n--- Starting Post Training Quantization ---")
    cpu_model_ptq = get_model_standard(current_model_name, num_classes=10).eval().to("cpu")
    if os.path.exists(weights_path):
        cpu_model_ptq.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

    qconfig_ptq = get_default_qconfig("fbgemm")
    cpu_model_ptq.qconfig = qconfig_ptq
    prepared_ptq_fx = prepare_fx(cpu_model_ptq, {"": qconfig_ptq}, example_inputs=cpu_input)

    if 'calibration_loader' in locals():
        with torch.no_grad():
            for image, _ in calibration_loader:
                prepared_ptq_fx(image.to("cpu", memory_format=torch.channels_last))

    quantized_model_cpu = convert_fx(prepared_ptq_fx)
    ptq_size = print_size_of_model(quantized_model_cpu, "PTQ", cpu_input)
    ptq_acc = evaluate_accuracy(quantized_model_cpu, test_loader, device="cpu")

    torch._dynamo.reset()
    with torch.no_grad():
        optimized_quant_cpu = torch.compile(quantized_model_cpu)
        optimized_quant_cpu(cpu_input)
        ptq_cpu_mu, ptq_cpu_std = estimate_latency(optimized_quant_cpu, cpu_input, "cpu")

    ptq_cuda_mu, ptq_cuda_std = 0.0, 0.0
    if torch.cuda.is_available():
        cuda_model_ptq = get_model_standard(current_model_name, num_classes=10).eval().to("cuda")
        if os.path.exists(weights_path):
            cuda_model_ptq.load_state_dict(torch.load(weights_path, map_location="cuda", weights_only=True))
        quantize_(cuda_model_ptq, int8_dynamic_activation_int8_weight())
        
        torch._dynamo.reset()
        with torch.no_grad():
            optimized_quant_cuda = torch.compile(cuda_model_ptq)
            optimized_quant_cuda(cuda_input) 
            ptq_cuda_mu, ptq_cuda_std = estimate_latency(optimized_quant_cuda, cuda_input, "cuda")

    global_results_summary.append((current_model_name, "Post Training Quantization", ptq_size, ptq_cuda_mu, ptq_cpu_mu, ptq_acc, ptq_cuda_std, ptq_cpu_std))

    # ------------------------------------------
    # 3. QUANTIZATION-AWARE TRAINING (QAT)
    # ------------------------------------------
    print("\n--- Starting Quantization Aware Training ---")
    cpu_model_qat = get_model_standard(current_model_name, num_classes=10).train().to("cpu")
    if os.path.exists(weights_path):
        cpu_model_qat.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

    qconfig_qat = get_default_qat_qconfig("fbgemm")
    cpu_model_qat.qconfig = qconfig_qat
    prepared_model_qat = prepare_qat_fx(cpu_model_qat, {"": qconfig_qat}, example_inputs=cpu_input)

    optimizer = optim.Adam(prepared_model_qat.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    batch_count = 0
    if 'calibration_loader' in locals():
        for images, labels in calibration_loader:
            images, labels = images.to("cpu", memory_format=torch.channels_last), labels.to("cpu")
            optimizer.zero_grad()
            loss = criterion(prepared_model_qat(images), labels)
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count >= 10: break

    prepared_model_qat.eval()
    quantized_model_qat = convert_fx(prepared_model_qat)
    qat_size = print_size_of_model(quantized_model_qat, "QAT", cpu_input)
    qat_acc = evaluate_accuracy(quantized_model_qat, test_loader, device="cpu")

    torch._dynamo.reset()
    with torch.no_grad():
        optimized_qat_cpu = torch.compile(quantized_model_qat)
        optimized_qat_cpu(cpu_input)
        qat_cpu_mu, qat_cpu_std = estimate_latency(optimized_qat_cpu, cpu_input, "cpu")

    qat_cuda_mu, qat_cuda_std = 0.0, 0.0
    if torch.cuda.is_available():
        cuda_model_qat = get_model_standard(current_model_name, num_classes=10).eval().to("cuda")
        cuda_model_qat.load_state_dict(cpu_model_qat.state_dict())
        quantize_(cuda_model_qat, int8_dynamic_activation_int8_weight())
        
        torch._dynamo.reset()
        with torch.no_grad():
            optimized_qat_cuda = torch.compile(cuda_model_qat)
            optimized_qat_cuda(cuda_input) 
            qat_cuda_mu, qat_cuda_std = estimate_latency(optimized_qat_cuda, cuda_input, "cuda")

    global_results_summary.append((current_model_name, "Quantization Aware Training", qat_size, qat_cuda_mu, qat_cpu_mu, qat_acc, qat_cuda_std, qat_cpu_std))

# ==========================================
# FINAL SUMMARY TABLE FORMAT (GROUPED BY MODEL)
# ==========================================
print("\n")
print("="*105)
print("FINAL BENCHMARKING RESULTS")
print("="*105)

# Group the results by model
grouped_results = {model: [] for model in models_to_test}
for row in global_results_summary:
    grouped_results[row[0]].append(row)

for model_name in models_to_test:
    print(f"\n### {display_names_map[model_name]}")
    print(f"| {'Method':<30} | {'Model Size':<12} | {'Accuracy':<10} | {'GPU Latency (ms)':<20} | {'CPU Latency (ms)':<20} |")
    print(f"|{'-'*32}|{'-'*14}|{'-'*12}|{'-'*22}|{'-'*22}|")
    
    for _, method, size, cuda_l, cpu_l, acc, cuda_s, cpu_s in grouped_results[model_name]:
        cpu_str = f"{cpu_l:.2f} ± {cpu_s:.2f} ms" if cpu_l > 0 else "N/A"
        cuda_str = f"{cuda_l:.2f} ± {cuda_s:.2f} ms" if cuda_l > 0 else "N/A"
        size_str = f"{size:.2f} MB"
        acc_str = f"{acc:.2f}%"
        
        print(f"| {method:<30} | {size_str:<12} | {acc_str:<10} | {cuda_str:<20} | {cpu_str:<20} |")
print("\n")