import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

# Per-Class Augmentation
left_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

right_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

down_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

up_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

stop_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

zero_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Load Dataset
root = "dataset_retrain"
dataset = datasets.ImageFolder(root)
class_names = dataset.classes
print("[INFO] Classes:", class_names)

def per_class_transform(img, label):
    cname = class_names[label]
    if cname == "left":  return left_transform(img)
    if cname == "right": return right_transform(img)
    if cname == "down":  return down_transform(img)
    if cname == "up":    return up_transform(img)
    if cname == "stop":  return stop_transform(img)
    if cname == "zero":  return zero_transform(img)
    return test_transform(img)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, imagefolder):
        self.imagefolder = imagefolder

    def __len__(self):
        return len(self.imagefolder)

    def __getitem__(self, idx):
        img, label = self.imagefolder[idx]
        return per_class_transform(img, label), label

full_dataset = CustomDataset(dataset)

# Train/Val/Test Split
train_len = int(0.7 * len(full_dataset))
val_len   = int(0.15 * len(full_dataset))
test_len  = len(full_dataset) - train_len - val_len

train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len])

def val_test_wrapper(ds):
    class Wrapper(torch.utils.data.Dataset):
        def __len__(self): return len(ds)
        def __getitem__(self, i):
            img, label = dataset[ds.indices[i]]
            return test_transform(img), label
    return Wrapper()

val_ds  = val_test_wrapper(val_ds)
test_ds = val_test_wrapper(test_ds)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False)


# Build Model
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

EPOCHS = 50
best_acc = 0

train_acc_history = []
val_acc_history   = []
train_loss_history = []
val_loss_history   = []
val_f1_history     = []

for epoch in range(EPOCHS):

    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(out, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = running_loss / len(train_loader)

    train_acc_history.append(train_acc)
    train_loss_history.append(train_loss)

    model.eval()
    val_correct = 0
    val_total = 0
    val_running_loss = 0
    val_all_preds = []
    val_all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            
            loss = criterion(out, labels)
            val_running_loss += loss.item()

            _, preds = torch.max(out, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            val_all_preds.extend(preds.cpu().numpy())
            val_all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total
    val_loss = val_running_loss / len(val_loader)
    val_f1 = f1_score(val_all_labels, val_all_preds, average="macro")

    val_acc_history.append(val_acc)
    val_loss_history.append(val_loss)
    val_f1_history.append(val_f1)

    print(f"Epoch {epoch+1}/{EPOCHS}  "
          f"Train Acc={train_acc:.4f}  Val Acc={val_acc:.4f}  "
          f"Train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}  "
          f"Val F1={val_f1:.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model_perclass.pth")
        print(">>> Saved BEST model!")

    scheduler.step()

model.load_state_dict(torch.load("best_model_perclass.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        _, preds = torch.max(out, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# Accuracy Curve
plt.figure(figsize=(8,5))
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training / Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Loss Curve
plt.figure(figsize=(8,5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# F1 Curve
plt.figure(figsize=(8,5))
plt.plot(val_f1_history, label="Validation F1 (Macro)", color='purple')
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Validation F1 Score Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Final Metrics
test_f1 = f1_score(all_labels, all_preds, average="macro")
print("\nFINAL TEST ACCURACY:", best_acc)
print("FINAL TEST F1 SCORE:", test_f1)
print("Model saved as: best_model_perclass.pth")
