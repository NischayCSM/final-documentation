import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_transforms=transforms.Compose([
     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = r"C:\Users\nisch\OneDrive\Documents\paddy-disease-classification\train_images"
    
    train_full = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    val_full = datasets.ImageFolder(root=data_dir, transform=val_transforms)
    
    class_names = train_full.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    csv_path=r"C:\Users\nisch\OneDrive\Documents\paddy-disease-classification\train.csv"
    train_df = pd.read_csv(csv_path)
    print(train_df.shape)
    print(train_df.label.value_counts())
    
    targets = train_full.targets
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
            test_size=0.2,
            random_state=42,
            stratify=targets
        )
    
    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(val_full, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    num_epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
        
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            loss.backward()
            optimizer.step()

            # Accumulate metrics for ALL images in the epoch
            running_loss += loss.item() * images.size(0)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
    end_time = time.time()

    # 3. Calculate and format the duration
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    
    print(f"\nTraining completed in: {minutes}m {seconds}s")
            
    torch.save(model.state_dict(), "efficientnet_paddy.pth")
    print("\nModel saved to efficientnet_paddy.pth")

    generate_report_and_matrix(model,val_loader, class_names, device)

def generate_report_and_matrix(model, val_loader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (MobileNetV2)")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=5))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=False
    )
    plt.title('Confusion matrix', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_efficientnet.png", dpi=300)
    plt.show()

if __name__=="__main__":
    main()   