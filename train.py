import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Import our new dataset module
from dataset import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Universal Paddy Disease Training Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training images")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["resnet18", "densenet121", "efficientnet_b0", "mobilenet_v2", "vgg16", "googlenet"],
                        help="Select the model architecture to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()

def build_model(model_name, num_classes):
    """Factory function to initialize models and modify their classifier heads."""
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "googlenet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Training Model: {args.model}")

    # 1. Load Data (from our modularized dataset.py)
    train_loader, val_loader, class_names = get_dataloaders(args.data_dir, args.batch_size)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes.")

    # 2. Build Model
    model = build_model(args.model, num_classes).to(device)

    # 3. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training Loop
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            
            # Handle GoogLeNet auxiliary outputs during training
            if args.model == "googlenet" and isinstance(outputs, models.GoogLeNetOutputs):
                loss = criterion(outputs.logits, labels) + 0.3 * criterion(outputs.aux_logits2, labels) + 0.3 * criterion(outputs.aux_logits1, labels)
                preds = torch.argmax(outputs.logits, dim=1)
            else:
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        
        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        print(f"Train Loss: {running_loss/total_train:.4f} | Train Acc: {correct_train/total_train:.4f} | "
              f"Val Loss: {val_loss/total_val:.4f} | Val Acc: {correct_val/total_val:.4f}")

    print(f"\nTraining completed in: {int(time.time() - start_time) // 60}m {int(time.time() - start_time) % 60}s")

    # 5. Save Artifacts
    weights_filename = f"{args.model}_paddy.pth"
    torch.save(model.state_dict(), weights_filename)
    print(f"Model saved to '{weights_filename}'")
    generate_report(model, val_loader, class_names, device, args.model)

def generate_report(model, val_loader, class_names, device, model_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print(f"\n{'='*60}\nCLASSIFICATION REPORT ({model_name.upper()})\n{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=5))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()