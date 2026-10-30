# 📊 Project Progress & Milestone Tracker

**Project Name:** FloraVision — Plant Leaf Disease Detection & Lesion Segmentation System  
**Category:** Computer Vision Mini Project (20 Marks | CO1 to CO5 Comprehensive)  
**Repository:** [https://github.com/Indra1012/FloraVision](https://github.com/Indra1012/FloraVision)  
**Last Updated:** 2026-10-30  
**Overall Completion:** `100%` ✅

---

## 📈 Status Dashboard

| Phase | Milestone Description | Syllabus CO | Status | Completion | Key Deliverables |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Phase 0** | Environment Setup & Scaffolding | — | 🟢 Completed | 100% | Python 3.13, PyTorch 2.15 CPU, OpenCV 5.0, Scaffolding |
| **Phase 1** | Multi-Plant Dataset (9 Crops, 33 Classes) | CO1 | 🟢 Completed | 100% | 40K raw images, 6,552 balanced split (70/15/15), 33 sample leaves |
| **Phase 2** | Classical Image Processing Suite | CO1, CO2 | 🟢 Completed | 100% | RGB $\to$ HSV, CLAHE, Bilateral, Sobel, Canny, ORB keypoints |
| **Phase 3** | Lesion Segmentation & Severity Metric | CO3 | 🟢 Completed | 100% | Otsu + HSV necrosis masking, % Affected Area formula, contour overlay |
| **Phase 4** | Deep Learning Models (CNN vs MobileNetV2) | CO3, CO4 | 🟢 Completed | 100% | Baseline CNN (73.98%) vs MobileNetV2 (**96.95%**), confusion matrices |
| **Phase 5** | Explainable AI via Grad-CAM Heatmaps | CO5 | 🟢 Completed | 100% | Gradient-weighted Class Activation Mapping on MobileNetV2 `features[-1]` |
| **Phase 6** | Diagnostic Ingestion & Validation Pipeline | CO1–CO5 | 🟢 Completed | 100% | Universal format support (JPG/PNG/WEBP/BMP), normalization, OOD filter |
| **Phase 7** | Apple-Style Glassmorphism Local Web UI | — | 🟢 Completed | 100% | Translucent blur cards, 9-crop subcategory selector, Top-3 bars, 3 tabs |
| **Phase 8** | Academic Documentation, PPT & GitHub | CO1–CO5 | 🟢 Completed | 100% | 3 Jupyter notebooks, Viva guide, report, pushed to GitHub repository |

---

## 📋 Granular Milestone Execution Log

### ⚙️ Phase 0: Setup & Scaffolding
- ✅ Initialized directory hierarchy: `data/`, `notebooks/`, `src/`, `models/`, `static/`, `templates/`, `reports/`, `scripts/`
- ✅ Created verified `requirements.txt` with PyTorch, Torchvision, OpenCV, Scikit-learn, NumPy, Matplotlib, Pillow
- ✅ Resolved Windows console encoding (`utf-8` reconfigure) and Matplotlib cache path handling
- ✅ Created `start.bat` for 1-click desktop launching

---

### 📂 Phase 1: Multi-Crop Dataset Acquisition & Balanced Partitioning
- ✅ Downloaded and curated PlantVillage multi-crop dataset covering **9 distinct plant species** across **33 disease and healthy classes**:
  1. **Apple:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy
  2. **Bell Pepper:** Bacterial Spot, Healthy
  3. **Cherry:** Powdery Mildew, Healthy
  4. **Corn (Maize):** Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
  5. **Grape:** Black Rot, Esca (Black Measles), Leaf Blight, Healthy
  6. **Peach:** Bacterial Spot, Healthy
  7. **Potato:** Early Blight, Late Blight, Healthy
  8. **Strawberry:** Leaf Scorch, Healthy
  9. **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl, Healthy
- ✅ Balanced dataset split with 200 images/class cap:
  - **Train Set (70%):** 4,586 images
  - **Validation Set (15%):** 982 images
  - **Test Set (15%):** 984 images
  - **Total Balanced:** 6,552 images
- ✅ Curated 33 dedicated sample leaves in `data/samples/` for instantaneous offline testing
- ✅ Generated visual charts: `reports/plant_species_distribution.png` and `reports/class_distribution.png`

---

### 🧪 Phase 2: Classical Image Processing Suite (CO1 & CO2)
- ✅ Implemented modular pipeline in `src/classical_processing.py`:
  - **Color conversions:** RGB $\to$ Grayscale, RGB $\to$ HSV (separating Hue for chlorophyll greens vs lesion browns)
  - **Contrast enhancement:** Contrast Limited Adaptive Histogram Equalization (**CLAHE**) on Luminance channel
  - **Spatial filtering:** Gaussian blur, Median filter, and edge-preserving Bilateral filtering
  - **Edge detection:** Sobel horizontal/vertical gradients and Canny multi-stage hysteresis edge detector
  - **Feature extraction:** ORB (Oriented FAST and Rotated BRIEF) invariant keypoint descriptors
- ✅ Saved 5 multi-panel comparative figures across crops in `reports/classical_*.png`

---

### 🔬 Phase 3: Lesion Segmentation & Severity Metric (CO3)
- ✅ Implemented dual-stage segmentation in `src/segmentation.py`:
  - **Stage 1 (Leaf Mask):** Otsu thresholding combined with saturation masking and morphological cleanup
  - **Stage 2 (Disease Lesion Mask):** HSV color thresholding isolating necrosis and chlorosis
- ✅ Implemented quantitative mathematical severity index:
  $$\text{Affected Leaf Area (\%)} = \frac{\text{Diseased Lesion Pixels}}{\text{Total Leaf Blade Pixels}} \times 100$$
- ✅ Automated health condition classification:
  - `Healthy`: $0\%$
  - `Mild Infection`: $< 10\%$
  - `Moderate Infection`: $10\% - 25\%$
  - `Severe Infection`: $> 25\%$
- ✅ Generated contour overlays with neon boundaries and translucent heat tints (`reports/segmentation_*.png`)

---

### 🧠 Phase 4: Deep Learning Classification (CO3 & CO4)
- ✅ Built robust PyTorch training pipeline with data augmentations in `src/train.py`
- ✅ **Model 1 (Baseline Custom CNN):**
  - 4 Convolutional blocks (Conv2D $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool) + 2 Linear layers with Dropout (0.5)
  - **Test Accuracy:** `73.98%` | **Precision:** `74.89%` | **Recall:** `74.04%` | **F1-Score:** `72.58%`
  - Saved weights: `models/baseline_custom_cnn_best.pth` (10.0 MB)
- ✅ **Model 2 (MobileNetV2 Transfer Learning):**
  - Pretrained ImageNet backbone with inverted residuals and depthwise separable convolutions
  - Customized classifier head: `Dropout(0.2) -> Linear(1280, 33)`
  - **Test Accuracy:** `96.95%` (+22.97% improvement!) | **Precision:** `97.02%` | **Recall:** `96.97%` | **F1-Score:** `96.96%`
  - Saved weights: `models/mobilenetv2_best.pth` (11.8 MB)
- ✅ Computed confusion matrices and saved reports in `reports/confusion_matrix_*.png` and `reports/model_comparison_metrics.txt`

---

### 🔍 Phase 5: Explainability & Model Interpretability (CO5)
- ✅ Implemented Grad-CAM (Gradient-weighted Class Activation Mapping) in `src/explainability.py`
- ✅ Hooked into final convolutional layer of MobileNetV2 (`model.features[-1]`)
- ✅ Generated spatial attention heatmaps proving the AI model focuses on actual necrotic lesions rather than background artifacts (`reports/gradcam_*.png`)

---

### 🛡️ Phase 6: Diagnostic Pipeline & Ingestion Engine
- ✅ Unified end-to-end engine in `src/inference.py`: accepts any image format, normalizes channels, runs classification, segmentation, and Grad-CAM
- ✅ Base64 encoding for instantaneous local browser visualization without disk write overhead

---

### 🖥️ Phase 7: Apple-Style Glassmorphism Web Interface
- ✅ Built Apple-inspired design in `static/css/glass.css` with `backdrop-filter: blur(26px)`, ambient glowing gradients, and SF Pro typography
- ✅ Responsive 2-column layout:
  - **Left Panel:** Drag-and-drop leaf upload + 9-crop selector with collapsible sub-category disease chips
  - **Right Panel:** Health Condition HUD, Affected Leaf Area %, Crop & Disease diagnosis banner, animated Top-3 prediction progress bars
- ✅ Interactive 3-Tab visualizer:
  - **Tab 1:** Original Leaf vs Grad-CAM Model Attention Heatmap
  - **Tab 2:** Isolated Leaf Blade, Isolated Necrotic Lesions, Lesion Contour Overlay
  - **Tab 3:** Classical CV Transforms (CLAHE, Canny, Sobel, ORB Keypoints)
- ✅ Built zero-dependency local web server (`server.py`) using Python's native `http.server` running on `http://127.0.0.1:8000`

---

### 📑 Phase 8: Academic Reports, Presentation Prep & GitHub
- ✅ Generated comprehensive Academic Project Report in `reports/PROJECT_REPORT.md`
- ✅ Formulated complete Viva Voce Q&A and Course Outcomes mapping in `reports/VIVA_AND_PRACTICAL_RECORD.md`
- ✅ Created 3 interactive Jupyter notebooks in `notebooks/`:
  - `01_classical_processing.ipynb` (CO1, CO2)
  - `02_segmentation_analysis.ipynb` (CO3)
  - `03_gradcam_explainability.ipynb` (CO5)
- ✅ Initialized Git repository, verified `.gitignore`, and pushed complete codebase, models, sample images, and documentation to GitHub at `https://github.com/Indra1012/FloraVision`
