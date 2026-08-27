import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def get_dataloaders(data_dir, batch_size=32):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")

    # 1. Define Transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Load Datasets
    train_full = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    val_full = datasets.ImageFolder(root=data_dir, transform=val_transforms)
    
    class_names = train_full.classes
    
    # 3. Stratified Split (80/20)
    targets = train_full.targets
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        random_state=42,
        stratify=targets
    )

    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(val_full, val_idx)

    # Dynamic worker allocation: safe for Windows, fast for Linux
    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, class_names