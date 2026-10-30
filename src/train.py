"""Deep Learning Model Training and Evaluation Pipeline (CO3, CO4, Practicals 7 & 8).
Trains Baseline CNN and MobileNetV2 with Data Augmentation,
computes Precision/Recall/F1, generates confusion matrices, and plots learning curves.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Safe stdout handling for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import BaselineLeafCNN, build_mobilenet_v2

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_data_transforms():
    """Build data augmentation and normalization transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_test_transform


def load_datasets(batch_size=32):
    """Load train, val, test datasets using ImageFolder."""
    train_tf, eval_tf = get_data_transforms()

    train_path = DATA_DIR / "train"
    val_path = DATA_DIR / "val"
    test_path = DATA_DIR / "test"

    train_dataset = datasets.ImageFolder(train_path, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_path, transform=eval_tf)
    test_dataset = datasets.ImageFolder(test_path, transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Save class mapping
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    class_map_file = MODELS_DIR / "class_indices.json"
    with open(class_map_file, "w") as f:
        json.dump(train_dataset.class_to_idx, f, indent=4)
    print(f"[+] Saved class index mapping to: {class_map_file} ({len(train_dataset.classes)} classes)")

    return train_loader, val_loader, test_loader, train_dataset.classes


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation or test set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    eval_loss = running_loss / total
    eval_acc = accuracy_score(all_labels, all_preds)
    return eval_loss, eval_acc, np.array(all_preds), np.array(all_labels)


def plot_learning_curves(history, model_name):
    """Plot and save training & validation loss and accuracy curves."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], "o-", color="#1976d2", label="Training Loss", lw=2)
    ax1.plot(epochs, history["val_loss"], "s--", color="#d32f2f", label="Validation Loss", lw=2)
    ax1.set_title(f"{model_name}: Loss Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Cross Entropy Loss", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy Curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", color="#2e7d32", label="Training Acc", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", color="#e65100", label="Validation Acc", lw=2)
    ax2.set_title(f"{model_name}: Accuracy Curve", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    chart_path = REPORTS_DIR / f"training_curves_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] Saved training curves: {chart_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot and save multiclass confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 14))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)

    # Simplified display labels for clean chart
    clean_labels = [c.replace("___", " ").replace("_", " ") for c in class_names]
    tick_marks = np.arange(len(clean_labels))
    plt.xticks(tick_marks, clean_labels, rotation=90, fontsize=7)
    plt.yticks(tick_marks, clean_labels, fontsize=7)

    plt.ylabel("True Class", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.tight_layout()

    cm_path = REPORTS_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix: {cm_path}")


def train_model(model, model_name, train_loader, val_loader, test_loader, class_names, epochs=5, lr=1e-3, device="cpu"):
    """Full training, validation, and testing routine for a model."""
    print(f"\n" + "=" * 65)
    print(f"[*] Training {model_name} on {str(device).upper()} ({epochs} Epochs)")
    print("=" * 65)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_best.pth"

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)

    total_time = time.time() - start_time
    print(f"\n[+] Training completed in {total_time/60:.2f} mins. Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"[+] Saved best model weights: {best_weights_path}")

    # Plot learning curves
    plot_learning_curves(history, model_name)

    # Evaluate on Unseen Test Set
    print(f"\n🧪 Evaluating {model_name} on Unseen Test Set...")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    print(f"[+] Final Unseen Test Accuracy: {test_acc*100:.2f}%")

    # Plot Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, class_names, model_name)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    return {
        "model_name": model_name,
        "test_acc": test_acc,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "training_time_min": round(total_time / 60, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Plant Disease Classifiers")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--model", type=str, default="both", choices=["both", "mobilenet", "baseline"],
                        help="Which model to train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using computational device: {device}")

    train_loader, val_loader, test_loader, class_names = load_datasets(batch_size=args.batch_size)
    num_classes = len(class_names)

    comparison_results = []

    # 1. Custom Baseline CNN (Practical 7)
    if args.model in ["both", "baseline"]:
        baseline_net = BaselineLeafCNN(num_classes=num_classes)
        res_baseline = train_model(
            baseline_net, "Baseline Custom CNN", train_loader, val_loader, test_loader,
            class_names, epochs=args.epochs, lr=1e-3, device=device
        )
        comparison_results.append(res_baseline)

    # 2. MobileNetV2 Transfer Learning (Practical 8)
    if args.model in ["both", "mobilenet"]:
        mobilenet = build_mobilenet_v2(num_classes=num_classes, pretrained=True, freeze_features=False)
        res_mobilenet = train_model(
            mobilenet, "MobileNetV2", train_loader, val_loader, test_loader,
            class_names, epochs=args.epochs, lr=3e-4, device=device
        )
        comparison_results.append(res_mobilenet)

    # Save Comparison Summary
    summary_txt = REPORTS_DIR / "model_comparison_metrics.txt"
    with open(summary_txt, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("      DEEP LEARNING MODEL COMPARISON (CO3, CO4, PRACTICALS 7 & 8)\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Model Name':<24} | {'Test Acc':<9} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Time (min)':<10}\n")
        f.write("-" * 75 + "\n")
        for res in comparison_results:
            f.write(f"{res['model_name']:<24} | {res['test_acc']*100:<8.2f}% | "
                    f"{res['macro_precision']*100:<8.2f}% | {res['macro_recall']*100:<7.2f}% | "
                    f"{res['macro_f1']*100:<7.2f}% | {res['training_time_min']:<10}\n")
        f.write("=" * 75 + "\n")

    print(f"\n[+] Saved comparative report to: {summary_txt}")
    with open(summary_txt) as f:
        print(f.read())


if __name__ == "__main__":
    main()

