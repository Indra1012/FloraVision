# 🌿 FloraVision: Plant Leaf Disease Detection & Lesion Segmentation System
### An End-to-End Computer Vision & Deep Learning Suite with Apple-Style Glass UI

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![Test Accuracy](https://img.shields.io/badge/MobileNetV2%20Accuracy-96.95%25-brightgreen?style=flat)](#-model-benchmarks--results)
[![Course Outcomes](https://img.shields.io/badge/Syllabus-CO1%20to%20CO5-blue?style=flat)](#-course-outcomes-co-mapping)
[![GitHub Repo](https://img.shields.io/badge/GitHub-FloraVision-181717?style=flat&logo=github&logoColor=white)](https://github.com/Indra1012/FloraVision)

> **Computer Vision Mini Project (20 Marks Practical Evaluation)**  
> Comprehensive end-to-end implementation covering **Course Outcomes CO1 through CO5** across **9 plant species** and **33 classes**.

---

## ⚡ Quick Start

### 1. Launch the Application
Simply double-click **`start.bat`** or run in your terminal:
```bash
python server.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

*(The server runs with Python's built-in `http.server` — zero external web framework dependencies required!)*

---

## 🏛️ System Highlights

- 🌿 **Multi-Plant Recognition:** Diagnoses diseases across **9 crops** (Apple, Bell Pepper, Cherry, Corn, Grape, Peach, Potato, Strawberry, Tomato) spanning **33 conditions**.
- 🧠 **High Diagnostic Accuracy:** Transfer Learning via **MobileNetV2** achieving **96.95% unseen test accuracy** (compared with custom baseline CNN at 73.98%).
- 🔬 **Quantitative Lesion Severity Metric:** Automatically computes the exact damaged surface area percentage:
  $$\text{Affected Leaf Area (\%)} = \frac{\text{Diseased Lesion Pixels}}{\text{Total Leaf Blade Pixels}} \times 100$$
- 🔍 **Explainable AI (Grad-CAM):** Generates gradient-weighted attention heatmaps showing the exact spatial features influencing the model's diagnosis.
- 🧪 **Classical Computer Vision Suite:** Features real-time CLAHE contrast enhancement, Bilateral filtering, Sobel gradients, Canny edge detection, and ORB invariant keypoints.
- 🍏 **Apple-Style Glassmorphism UI:** Features translucent frosted cards (`backdrop-filter: blur(26px)`), interactive 9-crop selector with collapsible sub-categories, animated Top-3 prediction progress bars, and a 3-tab visualizer.
- 📚 **Academic Jupyter Notebooks:** 3 ready-to-run notebooks in `notebooks/` structured for practical evaluation.

---

## 📊 Model Benchmarks & Results

| Model Architecture | Base Weights | Test Accuracy | Precision | Recall | F1-Score | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Custom CNN** | Scratch (4 Conv Blocks) | 73.98% | 74.89% | 74.04% | 72.58% | 10.0 MB |
| **MobileNetV2 (Ours)** | **ImageNet-1K Pretrained** | **96.95%** | **97.02%** | **96.97%** | **96.96%** | **11.8 MB** |

*MobileNetV2 achieved a **+22.97% accuracy gain** over the baseline custom CNN through inverted residual depthwise separable convolutions and transfer learning.*

---

## 🎯 Course Outcomes (CO) Mapping

| Course Outcome | Focus Area | Project Implementation |
| :--- | :--- | :--- |
| **CO1** | Digital Image Fundamentals | RGB to Grayscale, RGB to HSV decomposition, ImageNet normalization |
| **CO2** | Enhancement, Filtering & Edges | CLAHE equalization, Bilateral filter, Sobel gradients, Canny edges, ORB descriptors |
| **CO3** | Segmentation & Morphology | Otsu thresholding, HSV lesion masking, quantitative $\text{Severity \%}$ calculation |
| **CO4** | Deep Learning Classification | Custom CNN vs fine-tuned MobileNetV2 (96.95% Acc), confusion matrix analysis |
| **CO5** | Modern CV & Explainability | **Grad-CAM** gradient-weighted class activation heatmaps overlaid on leaves |

---

## 📂 Project Structure

```text
CV_final_project/
├── data/
│   ├── raw/                      # Downloaded PlantVillage subset (40,000 images)
│   ├── processed/                # Balanced train, val, test splits (6,552 images)
│   └── samples/                  # Curated sample leaves across all 9 crops
├── notebooks/                    # Academic lab submission notebooks
│   ├── 01_classical_processing.ipynb   # CO1, CO2 (Color, Filtering, Edges, ORB)
│   ├── 02_segmentation_analysis.ipynb   # CO3 (Otsu, HSV Lesions, Severity %)
│   └── 03_gradcam_explainability.ipynb # CO5 (MobileNetV2 Grad-CAM Heatmaps)
├── src/                          # Modular production python package
│   ├── classical_processing.py   # Grayscale, HSV, CLAHE, Sobel, Canny, ORB
│   ├── segmentation.py           # Background isolation, lesion masking, area %
│   ├── models.py                 # Baseline CNN & MobileNetV2 architectures
│   ├── train.py                  # PyTorch training loop, augmentations, metrics
│   ├── explainability.py         # Grad-CAM attention heatmap engine
│   └── inference.py              # Unified diagnostic inference pipeline
├── models/                       # Trained model weights & labels
│   ├── baseline_custom_cnn_best.pth  (73.98% Test Acc)
│   ├── mobilenetv2_best.pth          (96.95% Test Acc)
│   └── class_indices.json
├── static/                       # Web frontend assets (Apple Glass design)
│   ├── css/glass.css             # Frosted glassmorphism, blur, neon badges
│   └── js/app.js                 # Drag-and-drop, 9-crop selector & API handlers
├── templates/
│   └── index.html                # Apple glass aesthetic layout
├── reports/                      # Academic report, figures, viva guide
│   ├── PROJECT_REPORT.md         # Full Academic Mini Project Report
│   ├── VIVA_AND_PRACTICAL_RECORD.md # Comprehensive Viva Voce Q&A guide
│   ├── model_comparison_metrics.txt # Accuracy, Precision, Recall, F1 table
│   └── *.png                     # Confusion matrices, curves, distribution charts
├── server.py                     # Zero-dependency local web server
├── start.bat                     # 1-Click desktop launcher
├── PROGRESS.md                   # 100% milestone tracker & execution log
└── PROJECT_OVERVIEW.md           # Syllabus, architecture, and PPT slide guide
```

---

## 🖥️ Web Interface Features

The user interface accessible at `http://127.0.0.1:8000` includes:
1. **Interactive 9-Plant Selector:** Apple, Pepper, Cherry, Corn, Grape, Peach, Potato, Strawberry, Tomato. Clicking any plant expands its disease sub-categories for instant 1-click testing.
2. **Health Condition HUD:** Real-time health condition badge (`Healthy`, `Mild Infection`, `Moderate Infection`, `Severe Infection`).
3. **Quantitative Severity Index:** Displays the exact mathematical percentage of leaf blade affected by necrosis.
4. **Diagnosis Banner:** Clear crop species tag and disease diagnosis with animated Top-3 prediction progress bars.
5. **Multi-Tab Visualizer:**
   - **Tab 1: AI & Grad-CAM** (Original leaf vs Grad-CAM model attention heatmap)
   - **Tab 2: Lesion Segmentation** (Isolated leaf blade, isolated necrotic spots, and contour overlay)
   - **Tab 3: Classical CV** (CLAHE enhancement, Canny edges, Sobel magnitude, and ORB keypoints)

---

## 📑 Academic Documentation & Report Preparation

All files are structured to directly assist in preparing your project report and presentation:
- 📄 [**Full Academic Project Report** (`reports/PROJECT_REPORT.md`)](reports/PROJECT_REPORT.md)
- 🎓 [**Viva Voce & Practical Record Guide** (`reports/VIVA_AND_PRACTICAL_RECORD.md`)](reports/VIVA_AND_PRACTICAL_RECORD.md)
- 📊 [**Slide-by-Slide PPT Presentation Guide** (`PROJECT_OVERVIEW.md#7-ppt-presentation-slide-by-slide-guide`)](PROJECT_OVERVIEW.md)
- 📈 [**Milestone Tracker & Changelog** (`PROGRESS.md`)](PROGRESS.md)

---

## 👨‍💻 Author & Repository

- **Author:** Mahindra ([@Indra1012](https://github.com/Indra1012))
- **GitHub Repository:** [https://github.com/Indra1012/FloraVision](https://github.com/Indra1012/FloraVision)
