# 📊 Project Progress & Milestone Tracker

**Project Name:** Plant Leaf Disease Detection & Segmentation System (FloraVision)  
**Target CIA:** Labs / Mini Project Practical (20 Marks | CO1 to CO5)  
**Last Updated:** 2026-10-30  
**Overall Completion:** `100%`

---

## 📈 Status Dashboard

| Phase | Description | Syllabus CO | Status | Completion |
| :---: | :--- | :---: | :---: | :---: |
| **Phase 0** | Environment Setup & Project Scaffolding | — | 🟢 Completed | 100% |
| **Phase 1** | Dataset Acquisition & Preprocessing Split | CO1 | 🟢 Completed | 100% |
| **Phase 2** | Classical Image Processing (Color, Filter, Edges, ORB) | CO1, CO2 | 🟢 Completed | 100% |
| **Phase 3** | Leaf & Lesion Segmentation, Area % Metric | CO3 | 🟢 Completed | 100% |
| **Phase 4** | Deep Learning Classification (CNN & MobileNetV2) | CO3, CO4 | 🟢 Completed | 100% |
| **Phase 5** | Model Interpretability & Grad-CAM Visualizer | CO5 | 🟢 Completed | 100% |
| **Phase 6** | Robust Upload & In-the-Wild Pipeline | CO1–CO5 | 🟢 Completed | 100% |
| **Phase 7** | Apple-Style Glassmorphism Web Interface | — | 🟢 Completed | 100% |
| **Phase 8** | Final Academic Report & Demo Preparation | CO1–CO5 | 🟢 Completed | 100% |

*Legend: ⚪ Pending | 🟡 In Progress | 🟢 Completed | 🔴 Blocked*

---

## 📋 Detailed Phased Action Items

### ⚙️ Phase 0: Setup & Scaffolding
- ✅ Create `PROJECT_OVERVIEW.md` with complete architectural and syllabus mapping
- ✅ Create `PROGRESS.md` progress tracking file
- ✅ Initialize directory hierarchy (`data/`, `notebooks/`, `src/`, `models/`, `static/`, `templates/`, `reports/`)
- ✅ Create `requirements.txt` with verified dependency versions
- ✅ Initialize and test local Python environment with complete scientific stack
- ✅ Verify CPU/GPU acceleration and OpenCV 5.0 / PyTorch 2.15 installation

---

### 📂 Phase 1: Dataset Acquisition & Partitioning
- ✅ Expanded dataset to **9 plant species** (Apple, Bell Pepper, Cherry, Corn, Grape, Peach, Potato, Strawberry, Tomato) with **33 classes**
- ✅ Automated dataset downloader & sparse-checkout script (40,000 raw images in `data/raw/`)
- ✅ Balanced train/val/test split (`70% Train: 4,586`, `15% Val: 982`, `15% Test: 984`)
- ✅ Generated plant species chart (`reports/plant_species_distribution.png`) and 33-class chart (`reports/class_distribution.png`)
- ✅ Curated sample test images for all 33 classes into `data/samples/`

---

### 🧪 Phase 2: Classical Image Processing (CO1 & CO2)
- ✅ Implement color space conversion module (`RGB` $\to$ `Grayscale`, `RGB` $\to$ `HSV`) in `src/classical_processing.py`
- ✅ Implement histogram equalization (Standard & CLAHE on Luminance channel)
- ✅ Implement spatial smoothing filters (Gaussian, Median, and edge-preserving Bilateral)
- ✅ Implement edge detection pipelines (Sobel gradient magnitude & Canny edge detector)
- ✅ Implement ORB (Oriented FAST and Rotated BRIEF) feature keypoint detector
- ✅ Generate comparative before/after multi-panel figures for academic report across multiple plant species (`reports/classical_*.png`)

---

### 🔬 Phase 3: Segmentation & Severity Metric (CO3)
- ✅ Implement leaf-background segmentation mask (Color masking + Otsu thresholding + Morphology) in `src/segmentation.py`
- ✅ Implement diseased spot isolation (HSV necrosis/chlorosis thresholding)
- ✅ Calculate quantitative metric: $\text{Affected Area \%} = \frac{\text{Lesion Pixels}}{\text{Leaf Pixels}} \times 100$
- ✅ Create visual overlay function (neon contours & heat tint highlighting diseased spots on leaf)
- ✅ Generate multi-plant comparison figures and summary chart (`reports/segmentation_*.png`)

---

### 🧠 Phase 4: Deep Learning Classification (CO3 & CO4)
- ✅ Build dataset loaders with PyTorch data augmentation (flips, affine rotation, color jitter) in `src/train.py`
- ✅ Develop Baseline Custom CNN (3 conv-pool blocks) in `src/models.py` to demonstrate basic convolution principles (73.98% Test Acc)
- ✅ Implement Transfer Learning with **MobileNetV2** (Pretrained ImageNet base) in `src/models.py` (96.95% Test Acc)
- ✅ Train baseline CNN and save model weights (`models/baseline_custom_cnn_best.pth`)
- ✅ Fine-tune MobileNetV2 and save model weights (`models/mobilenetv2_best.pth`)
- ✅ Compute metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix (`reports/confusion_matrix_*.png`)
- ✅ Generate comparative performance summary table (Baseline CNN vs MobileNetV2) in `reports/model_comparison_metrics.txt`

---

### 🔍 Phase 5: Modern Techniques & Explainability (CO5)
- ✅ Implement Grad-CAM (Gradient-weighted Class Activation Mapping) on MobileNetV2 in `src/explainability.py`
- ✅ Generate heatmaps highlighting spatial regions influencing the model's prediction
- ✅ Build side-by-side visualizer: `[Original Image | Lesion Segmentation | Grad-CAM Heatmap]` saved in `reports/gradcam_*.png`

---

### 🛡️ Phase 6: Robust Upload & In-the-Wild Pipeline
- ✅ Universal image ingestion supporting `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` in `src/inference.py`
- ✅ Handle alpha channels, grayscale, and ImageNet color normalization
- ✅ Implement out-of-distribution / non-leaf sanity validation with `< 50%` confidence threshold warning
- ✅ Curate test sample images for quick offline demonstration across 33 classes in `data/samples/`

---

### 🖥️ Phase 7: Apple-Style Glassmorphism Local UI
- ✅ Build modern Apple Glass UI (`backdrop-filter: blur`, translucent panels, rounded cards) in `static/css/glass.css`
- ✅ Implement file drag-and-drop zone with instant preview in `templates/index.html` and `static/js/app.js`
- ✅ Display Top-3 classification results with animated progress bars
- ✅ Display quantitative metrics card: Health Status, Disease Name, % Leaf Area Affected
- ✅ Display multi-view visualizer tabs:
  - Tab 1: AI Prediction & Grad-CAM Heatmap
  - Tab 2: Segmentation & Disease Contours
  - Tab 3: Classical Processing (HSV, CLAHE, Canny Edges, Sobel, ORB)
- ✅ Connect frontend to local backend server (`server.py`) running at `http://127.0.0.1:8000`

---

### 📑 Phase 8: Academic Report & Viva Preparation
- ✅ Generate all high-resolution figures for practical record & mini project report in `reports/`
- ✅ Formulate Course Outcomes (CO1 to CO5) mapping document in `reports/VIVA_AND_PRACTICAL_RECORD.md`
- ✅ Prepare comprehensive Viva Voce questions & answers covering classical vs deep learning trade-offs
- ✅ Create 3 interactive step-by-step academic Jupyter notebooks in `notebooks/`

---

## 📝 Activity & Changelog

| Date | Phase | Action Taken | Author / Notes |
| :---: | :---: | :--- | :--- |
| **2026-10-30** | Phase 0 | Initialized project repository and created `PROJECT_OVERVIEW.md` detailing full syllabus alignment (CO1–CO5), architecture, and execution plan. | Antigravity |
| **2026-10-30** | Phase 0 | Created `PROGRESS.md` live tracking dashboard with granular milestones and status metrics. | Antigravity |
| **2026-10-30** | Phase 1 | Downloaded 18,160 PlantVillage tomato images (10 classes), generated class distribution chart, created balanced train/val/test splits (70/15/15), and curated sample images. | Antigravity |
| **2026-10-30** | Phase 1 | Expanded to **9 distinct plant species** (Apple, Bell Pepper, Cherry, Corn, Grape, Peach, Potato, Strawberry, Tomato) with **33 classes** (40,000 raw images). Generated species and class distribution charts, and created balanced multi-plant splits (6,552 images: 4,586 train, 982 val, 984 test). | Antigravity |
| **2026-10-30** | Phase 2 | Implemented modular classical processing pipeline (`src/classical_processing.py`) covering CO1 (RGB to Grayscale, HSV) and CO2 (CLAHE, Bilateral filtering, Sobel/Canny edge detection, ORB keypoints). Generated 5 high-resolution multi-panel academic comparison figures across multiple crops in `reports/`. | Antigravity |
| **2026-10-30** | Phase 3 | Implemented leaf segmentation and lesion isolation engine (`src/segmentation.py`). Calculated quantitative surface area % affected and health categories. Generated individual crop figures and a multi-crop summary comparison chart (`reports/segmentation_multi_plant_summary.png`). | Antigravity |
| **2026-10-30** | Phase 4 | Trained Baseline CNN (73.98% Test Acc) and MobileNetV2 Transfer Learning (96.95% Test Acc) across 33 classes. Saved weights and comparative performance table (`reports/model_comparison_metrics.txt`). | Antigravity |
| **2026-10-30** | Phase 5 | Implemented Grad-CAM explainability module (`src/explainability.py`) and generated attention heatmaps across multiple crops. | Antigravity |
| **2026-10-30** | Phase 6 & 7 | Developed unified diagnostic pipeline (`src/inference.py`), Apple-Style Glassmorphism UI (`static/css/glass.css`, `templates/index.html`), and launched local server (`server.py`) at `http://127.0.0.1:8000`. | Antigravity |
| **2026-10-30** | Phase 8 | Generated complete Viva Voce guide (`reports/VIVA_AND_PRACTICAL_RECORD.md`) and 3 interactive academic Jupyter notebooks in `notebooks/`. | Antigravity |
