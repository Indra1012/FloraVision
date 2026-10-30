# 🌿 FloraVision: Plant Leaf Disease Detection & Segmentation System
### An End-to-End Computer Vision & Deep Learning Suite with Apple-Style Glass UI

> **Computer Vision Mini Project (CIA Labs / Mini Project, 20 Marks)**  
> Comprehensive implementation covering **Course Outcomes CO1 through CO5** across **9 plant species** and **33 classes**.

---

## ⚡ Quick Start

### 1. Launch the Application
Simply double-click **`start.bat`** or run:
```bash
python server.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## 🏛️ System Features
- **Multi-Plant Recognition:** Detects diseases across **9 crops** (Apple, Bell Pepper, Cherry, Corn, Grape, Peach, Potato, Strawberry, Tomato) spanning **33 conditions**.
- **High Diagnostic Accuracy:** Transfer learning with **MobileNetV2** achieving **96.95% unseen test accuracy** (compared with custom baseline CNN at 73.98%).
- **Quantitative Severity Metric:** Computes the mathematical surface area percentage of leaf tissue affected by necrosis.
- **Explainability (Grad-CAM):** Displays gradient-weighted attention heatmaps showing where the model focused.
- **Apple Glassmorphism UI:** Translucent frosted cards (`backdrop-filter: blur`), glowing mesh gradients, animated top-3 confidence bars, and interactive multi-tab visualizers.
- **Academic Notebooks:** 3 interactive Jupyter notebooks in `notebooks/` ready for practical evaluation.

---

## 📂 Project Structure

```text
CV_final_project/
├── data/
│   ├── raw/                      # 40,000 raw PlantVillage multi-crop images
│   ├── processed/                # Balanced train, val, test splits (6,552 images)
│   └── samples/                  # Curated demo leaves across all 9 crops
├── notebooks/                    # Academic lab submission notebooks
│   ├── 01_classical_processing.ipynb   # CO1, CO2 (Practicals 2-4)
│   ├── 02_segmentation_analysis.ipynb   # CO3 (Practical 5)
│   └── 03_gradcam_explainability.ipynb # CO5 (Practicals 9 & 11)
├── src/                          # Modular production python package
│   ├── classical_processing.py   # Grayscale, HSV, CLAHE, Sobel, Canny, ORB
│   ├── segmentation.py           # Background isolation, lesion masking, area %
│   ├── models.py                 # Baseline CNN & MobileNetV2 architectures
│   ├── train.py                  # Training loop, metrics, confusion matrices
│   ├── explainability.py         # Grad-CAM attention heatmap engine
│   └── inference.py              # Unified diagnostic pipeline
├── models/                       # Trained model weights
│   ├── baseline_custom_cnn_best.pth (73.98% Test Acc)
│   ├── mobilenetv2_best.pth         (96.95% Test Acc)
│   └── class_indices.json
├── static/                       # Web frontend assets
│   ├── css/glass.css             # Apple-style glassmorphism styling
│   └── js/app.js                 # Drag-and-drop & API interaction controller
├── templates/
│   └── index.html                # Apple glass aesthetic layout
├── reports/                      # Academic report, figures, viva guide
│   ├── PROJECT_REPORT.md         # Full Academic Mini Project Report
│   ├── VIVA_AND_PRACTICAL_RECORD.md # Comprehensive Viva Voce Q&A guide
│   ├── model_comparison_metrics.txt # Accuracy, Precision, Recall, F1 table
│   └── *.png                     # Confusion matrices, curves, distribution charts
├── server.py                     # Zero-dependency local web server
├── start.bat                     # 1-Click desktop launcher
├── PROGRESS.md                   # 100% milestone tracker
└── PROJECT_OVERVIEW.md           # Syllabus & architectural specification
```

---

## 🎓 Academic Documentation
- 📄 [**Full Academic Project Report** (`reports/PROJECT_REPORT.md`)](file:///e:/CV_final%20project/reports/PROJECT_REPORT.md)
- 🎓 [**Viva Voce & Practical Record Guide** (`reports/VIVA_AND_PRACTICAL_RECORD.md`)](file:///e:/CV_final%20project/reports/VIVA_AND_PRACTICAL_RECORD.md)
- 📊 [**Milestone Tracker** (`PROGRESS.md`)](file:///e:/CV_final%20project/PROGRESS.md)

