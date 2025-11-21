import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import os
import numpy as np

# Train the model on the RGB dataset from Kaggle

# Use strong (direction-safe) augmentations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomRotation(20),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.15, 0.15),
        scale=(0.9, 1.05)
    ),
    transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load the dataset
dataset = datasets.ImageFolder("data/Dataset_RGB")

num_classes = len(dataset.classes)
print(f"[INFO] Detected {num_classes} gesture classes:", dataset.classes)

# 70/15/15 split
total = len(dataset)
train_len = int(0.7 * total)
val_len = int(0.15 * total)
test_len = total - train_len - val_len

train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

train_ds.dataset.transform = train_transform
val_ds.dataset.transform = val_test_transform
test_ds.dataset.transform = val_test_transform

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

print(f"[INFO] Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

# Build the MobileNetV3 model
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
EPOCHS = 50
best_val_acc = 0
patience = 7
patience_counter = 0

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0
    train_correct = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (out.argmax(1) == labels).sum().item()

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            val_correct += (out.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_ds)
    val_acc = val_correct / len(val_ds)

    print(f"Epoch [{epoch+1}/{EPOCHS}]  "
          f"Train Loss: {train_loss/len(train_loader):.4f}  "
          f"Train Acc: {train_acc:.4f}  "
          f"Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_mobilenetv3_augmented.pth")
        print(">>> Saved new BEST model!")
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= patience:
        print("[EARLY STOPPING] No improvement for several epochs.")
        break

    scheduler.step()

# Test the model
model.load_state_dict(torch.load("best_mobilenetv3_augmented.pth"))
model.eval()

test_correct = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        test_correct += (out.argmax(1) == labels).sum().item()

final_acc = test_correct / len(test_ds)
print(f"\nFINAL TEST ACCURACY: {final_acc:.4f}")
print("Best model saved as: best_mobilenetv3_augmented.pth")
