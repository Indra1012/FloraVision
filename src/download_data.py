import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# Tomato classes in PlantVillage
TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def main():
    print("Testing download sources for PlantVillage dataset...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if git sparse-checkout is viable or direct zip
    print(f"Target raw directory: {RAW_DIR}")

if __name__ == "__main__":
    main()

