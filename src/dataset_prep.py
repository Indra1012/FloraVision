"""Multi-Plant Dataset preparation script for PlantVillage dataset.
Covers 9 distinct plant species and 33 disease/healthy classes.
Performs species & disease distribution analysis, visualization, and stratified train/val/test split.
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Safe stdout handling for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")

import matplotlib.pyplot as plt

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Clean human-readable display names for all 33 classes
CLASS_INFO = {
    # 1. Apple
    "Apple___Apple_scab": {"plant": "Apple", "disease": "Apple Scab"},
    "Apple___Black_rot": {"plant": "Apple", "disease": "Black Rot"},
    "Apple___Cedar_apple_rust": {"plant": "Apple", "disease": "Cedar Apple Rust"},
    "Apple___healthy": {"plant": "Apple", "disease": "Healthy"},

    # 2. Cherry
    "Cherry_(including_sour)___Powdery_mildew": {"plant": "Cherry", "disease": "Powdery Mildew"},
    "Cherry_(including_sour)___healthy": {"plant": "Cherry", "disease": "Healthy"},

    # 3. Corn (Maize)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"plant": "Corn", "disease": "Cercospora Leaf Spot"},
    "Corn_(maize)___Common_rust_": {"plant": "Corn", "disease": "Common Rust"},
    "Corn_(maize)___Northern_Leaf_Blight": {"plant": "Corn", "disease": "Northern Leaf Blight"},
    "Corn_(maize)___healthy": {"plant": "Corn", "disease": "Healthy"},

    # 4. Grape
    "Grape___Black_rot": {"plant": "Grape", "disease": "Black Rot"},
    "Grape___Esca_(Black_Measles)": {"plant": "Grape", "disease": "Esca (Black Measles)"},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"plant": "Grape", "disease": "Leaf Blight"},
    "Grape___healthy": {"plant": "Grape", "disease": "Healthy"},

    # 5. Peach
    "Peach___Bacterial_spot": {"plant": "Peach", "disease": "Bacterial Spot"},
    "Peach___healthy": {"plant": "Peach", "disease": "Healthy"},

    # 6. Pepper (Bell)
    "Pepper,_bell___Bacterial_spot": {"plant": "Bell Pepper", "disease": "Bacterial Spot"},
    "Pepper,_bell___healthy": {"plant": "Bell Pepper", "disease": "Healthy"},

    # 7. Potato
    "Potato___Early_blight": {"plant": "Potato", "disease": "Early Blight"},
    "Potato___Late_blight": {"plant": "Potato", "disease": "Late Blight"},
    "Potato___healthy": {"plant": "Potato", "disease": "Healthy"},

    # 8. Strawberry
    "Strawberry___Leaf_scorch": {"plant": "Strawberry", "disease": "Leaf Scorch"},
    "Strawberry___healthy": {"plant": "Strawberry", "disease": "Healthy"},

    # 9. Tomato
    "Tomato___Bacterial_spot": {"plant": "Tomato", "disease": "Bacterial Spot"},
    "Tomato___Early_blight": {"plant": "Tomato", "disease": "Early Blight"},
    "Tomato___Late_blight": {"plant": "Tomato", "disease": "Late Blight"},
    "Tomato___Leaf_Mold": {"plant": "Tomato", "disease": "Leaf Mold"},
    "Tomato___Septoria_leaf_spot": {"plant": "Tomato", "disease": "Septoria Leaf Spot"},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"plant": "Tomato", "disease": "Spider Mites"},
    "Tomato___Target_Spot": {"plant": "Tomato", "disease": "Target Spot"},
    "Tomato___Tomato_mosaic_virus": {"plant": "Tomato", "disease": "Mosaic Virus"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"plant": "Tomato", "disease": "Yellow Leaf Curl"},
    "Tomato___healthy": {"plant": "Tomato", "disease": "Healthy"},
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_class_counts():
    counts = {}
    for class_folder in RAW_DIR.iterdir():
        if class_folder.is_dir():
            files = [f for f in class_folder.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
            if files:
                counts[class_folder.name] = files
    return counts


def plot_distributions(counts):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Aggregate counts by plant species
    species_counts = defaultdict(int)
    for class_name, files in counts.items():
        info = CLASS_INFO.get(class_name, {"plant": class_name.split("___")[0]})
        species_counts[info["plant"]] += len(files)

    # Plot Species Chart
    plt.figure(figsize=(10, 5))
    plants = list(species_counts.keys())
    totals = list(species_counts.values())
    colors = ["#2e7d32", "#1976d2", "#e65100", "#7b1fa2", "#00796b", "#c2185b", "#5d4037", "#f57c00", "#388e3c"]
    bars = plt.bar(plants, totals, color=colors[:len(plants)], edgecolor="#222", alpha=0.85)
    plt.title("PlantVillage Dataset: Image Count Across 9 Plant Species", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Plant Species", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Images", fontsize=11, fontweight="bold")
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 100, f"{int(height):,}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    species_chart_path = REPORTS_DIR / "plant_species_distribution.png"
    plt.savefig(species_chart_path, dpi=300)
    plt.close()
    print(f"[+] Plant species distribution chart saved to: {species_chart_path}")

    # 2. Detailed Class Distribution Chart
    plt.figure(figsize=(14, 11))
    class_labels = [f"{CLASS_INFO[k]['plant']} - {CLASS_INFO[k]['disease']}" if k in CLASS_INFO else k for k in counts.keys()]
    class_vals = [len(v) for v in counts.values()]

    # Sort by count
    sorted_pairs = sorted(zip(class_labels, class_vals), key=lambda x: x[1])
    sorted_labels, sorted_vals = zip(*sorted_pairs)

    bars2 = plt.barh(sorted_labels, sorted_vals, color="#2e7d32", edgecolor="#1b5e20", alpha=0.85)
    plt.title("Distribution Across All 33 Plant Disease & Healthy Classes", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Images", fontsize=11, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars2:
        width = bar.get_width()
        plt.text(width + 20, bar.get_y() + bar.get_height() / 2, f"{int(width):,}",
                 va="center", ha="left", fontsize=8, fontweight="bold", color="#1b5e20")

    plt.tight_layout()
    class_chart_path = REPORTS_DIR / "class_distribution.png"
    plt.savefig(class_chart_path, dpi=300)
    plt.close()
    print(f"[+] Detailed 33-class distribution chart saved to: {class_chart_path}")


def create_splits(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, max_per_class=200, seed=42):
    random.seed(seed)
    counts = get_class_counts()

    # Reset processed dir
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    for split in ["train", "val", "test"]:
        for class_name in counts.keys():
            (PROCESSED_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    split_stats = {"train": 0, "val": 0, "test": 0}

    print(f"\n[+] Preparing Multi-Plant Splits (Train: {int(train_ratio*100)}%, Val: {int(val_ratio*100)}%, Test: {int(test_ratio*100)}%)...")
    if max_per_class:
        print(f"[*] Balanced sampling capped at {max_per_class} images per class (ensures zero class imbalance and fast laptop training).")

    # Group by plant for organized console output
    classes_by_plant = defaultdict(list)
    for class_name in counts.keys():
        plant = CLASS_INFO.get(class_name, {}).get("plant", "Other")
        classes_by_plant[plant].append(class_name)

    for plant, class_list in sorted(classes_by_plant.items()):
        print(f"\n  🌱 [{plant.upper()}] ({len(class_list)} classes):")
        for class_name in sorted(class_list):
            files = counts[class_name]
            shuffled = files.copy()
            random.shuffle(shuffled)

            # Save demo sample
            sample_dest = SAMPLES_DIR / f"{class_name}_sample.jpg"
            if not sample_dest.exists() and len(shuffled) > 0:
                shutil.copy2(shuffled[0], sample_dest)

            if max_per_class and len(shuffled) > max_per_class:
                shuffled = shuffled[:max_per_class]

            n_total = len(shuffled)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train_files = shuffled[:n_train]
            val_files = shuffled[n_train:n_train + n_val]
            test_files = shuffled[n_train + n_val:]

            for f in train_files:
                shutil.copy2(f, PROCESSED_DIR / "train" / class_name / f.name)
            for f in val_files:
                shutil.copy2(f, PROCESSED_DIR / "val" / class_name / f.name)
            for f in test_files:
                shutil.copy2(f, PROCESSED_DIR / "test" / class_name / f.name)

            split_stats["train"] += len(train_files)
            split_stats["val"] += len(val_files)
            split_stats["test"] += len(test_files)

            disease = CLASS_INFO.get(class_name, {}).get("disease", class_name)
            print(f"    • {disease:<24} -> Train: {len(train_files):<4} | Val: {len(val_files):<3} | Test: {len(test_files):<3} | Total: {n_total}")

    print(f"\n[+] Multi-Plant Dataset splitting complete!")
    print(f"   Total Classes: {len(counts)} across 9 plant species")
    print(f"   Train: {split_stats['train']:,} | Val: {split_stats['val']:,} | Test: {split_stats['test']:,} | Total: {sum(split_stats.values()):,}")
    print(f"   Sample demo images copied to: {SAMPLES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Multi-Plant PlantVillage Dataset")
    parser.add_argument("--max-per-class", type=int, default=200,
                        help="Maximum images per class for balanced fast training (default: 200)")
    args = parser.parse_args()

    print("[*] Scanning raw multi-plant dataset...")
    counts = get_class_counts()
    total_imgs = sum(len(v) for v in counts.values())
    print(f"Found {len(counts)} classes with a total of {total_imgs:,} images.")

    plot_distributions(counts)
    create_splits(max_per_class=args.max_per_class)
