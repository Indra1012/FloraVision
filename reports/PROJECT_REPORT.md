# 🌿 FloraVision: Intelligent Multi-Plant Leaf Disease Detection, Segmentation & Explainability System

**Course:** Computer Vision (Labs & Mini Project)  
**Evaluation:** CIA Practical Evaluation (20 Marks | Practical No. 11: Research Project)  
**Coverage:** Course Outcomes CO1, CO2, CO3, CO4, CO5  
**Submission Date:** October 2026  

---

## 📑 Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction & Problem Statement](#2-introduction--problem-statement)
3. [Course Outcomes (CO) & Syllabus Mapping](#3-course-outcomes-co--syllabus-mapping)
4. [Dataset Description & Preprocessing](#4-dataset-description--preprocessing)
5. [Methodology & Architecture](#5-methodology--architecture)
   - 5.1 [Classical Image Processing (CO1 & CO2)](#51-classical-image-processing-co1--co2)
   - 5.2 [Lesion Segmentation & Severity Metric (CO3)](#52-lesion-segmentation--severity-metric-co3)
   - 5.3 [Deep Learning Classification (CO3 & CO4)](#53-deep-learning-classification-co3--co4)
   - 5.4 [Model Interpretability via Grad-CAM (CO5)](#54-model-interpretability-via-grad-cam-co5)
6. [Experimental Results & Discussion](#6-experimental-results--discussion)
   - 6.1 [Quantitative Model Comparison (CNN vs. MobileNetV2)](#61-quantitative-model-comparison)
   - 6.2 [Segmentation & Severity Quantification Results](#62-segmentation--severity-quantification-results)
7. [System Implementation & Apple Glass UI](#7-system-implementation--apple-glass-ui)
8. [Limitations & In-the-Wild Challenges](#8-limitations--in-the-wild-challenges)
9. [Conclusion & Future Scope](#9-conclusion--future-scope)
10. [References](#10-references)

---

## 1. Abstract
Plant foliar diseases pose severe threats to agricultural yield and food security. Early, accurate detection and severity quantification are paramount for localized agronomic treatment. This mini project presents **FloraVision**, an end-to-end Computer Vision system that unifies classical digital image processing, morphological segmentation, lightweight deep learning transfer learning, and gradient-weighted explainability into a modern Apple-style Glassmorphism local web interface. The system analyzes **9 distinct plant crops** (Apple, Bell Pepper, Cherry, Corn, Grape, Peach, Potato, Strawberry, Tomato) across **33 disease and healthy conditions**. Evaluated on 6,552 balanced images, our fine-tuned **MobileNetV2** transfer learning model achieved an unseen test accuracy of **96.95%** (Precision: 97.02%, Recall: 96.97%, F1-Score: 96.96%), substantially outperforming a custom 4-block baseline CNN (73.98%). Furthermore, our segmentation engine computes the quantitative percentage of infected leaf surface area, and **Grad-CAM** generates visual heatmaps validating that neural decision boundaries align with actual biological lesions.

---

## 2. Introduction & Problem Statement
Visual inspection of agricultural foliage by human pathologists is subjective, labor-intensive, and inaccessible in remote regions. While deep learning models offer automated diagnostics, typical solutions suffer from:
1. **The "Black-Box" Problem:** Inability to explain why a neural network diagnosed a specific disease.
2. **Lack of Severity Quantification:** Most systems provide only a discrete classification label without estimating what percentage of the leaf is damaged.
3. **High Computational Demands:** Bulky architectures (VGG-16, ResNet-50) require GPU infrastructure unavailable in field settings.
4. **Poor UI/UX:** Clunky interfaces that fail to present multi-stage image processing intuitively.

**FloraVision** solves these challenges by providing:
- Real-time classification running entirely on standard laptop CPUs via **MobileNetV2**.
- Mathematical quantification of lesion area:
  $$\text{Leaf Disease Severity (\%)} = \frac{\text{Pixel Area of Diseased Lesions}}{\text{Total Surface Area of Leaf Blade}} \times 100$$
- Visual interpretability through **Grad-CAM (Gradient-weighted Class Activation Mapping)**.
- A zero-dependency Apple Glass web interface featuring live drag-and-drop analysis and curated test samples.

---

## 3. Course Outcomes (CO) & Syllabus Mapping

This project explicitly satisfies all five Course Outcomes listed in the Computer Vision syllabus:

| Syllabus Unit | Course Outcome | Practical | Project Implementation |
| :--- | :--- | :---: | :--- |
| **Unit I: Image Fundamentals** | **CO1:** Understand digital image representations, color models, transformations | Practical 2 | RGB to Grayscale, HSV decomposition separating chlorophyll Hue ($35^\circ-85^\circ$) from lesion Saturation and Value. |
| **Unit II: Enhancement & Filtering** | **CO2:** Spatial filtering, edge detection, invariant feature descriptors | Practicals 3, 4 | Contrast Limited Adaptive Histogram Equalization (**CLAHE**), Bilateral edge-preserving filtering, **Sobel** directional gradient magnitudes ($G_x, G_y$), **Canny** multi-stage edge detection, and **ORB** keypoint extraction. |
| **Unit III: Segmentation** | **CO3:** Image segmentation and morphological operations | Practical 5 | Combined Otsu and saturation thresholding for leaf blade isolation; HSV necrosis thresholding for lesion extraction; mathematical surface area % quantification. |
| **Unit IV: Deep Learning** | **CO4:** CNN architectures, feature extraction, transfer learning | Practicals 7, 8 | Custom 4-block Baseline CNN (Conv2D $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool) vs. fine-tuned **MobileNetV2** (Inverted residual blocks with depthwise separable convolutions). |
| **Unit V: Modern CV & Interpretability** | **CO5:** Advanced CV techniques & explainable AI | Practicals 9, 11 | **Grad-CAM** implementation backpropagating gradients of target class scores to the final convolutional feature maps (`features[-1]`). |

---

## 4. Dataset Description & Preprocessing
The system utilizes the standardized **PlantVillage** multi-crop repository. While typical lab assignments restrict training to a single plant species, our project expanded the scope to **9 distinct agricultural crops**:

1. **Apple** (Apple Scab, Black Rot, Cedar Apple Rust, Healthy)
2. **Bell Pepper** (Bacterial Spot, Healthy)
3. **Cherry** (Powdery Mildew, Healthy)
4. **Corn (Maize)** (Cercospora Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy)
5. **Grape** (Black Rot, Esca Black Measles, Leaf Blight, Healthy)
6. **Peach** (Bacterial Spot, Healthy)
7. **Potato** (Early Blight, Late Blight, Healthy)
8. **Strawberry** (Leaf Scorch, Healthy)
9. **Tomato** (Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Spot, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl, Healthy)

### Data Balancing & Partitioning
- **Raw Pool:** 40,000 raw images archived in `data/raw/`.
- **Balanced Sampling:** To eliminate severe class imbalance (where Yellow Leaf Curl had 5,300+ images while Mosaic Virus had only 373), balanced sampling was applied.
- **Dataset Partition:** Stratified split with fixed random seed ($42$):
  - **Training Set (70%):** 4,586 images
  - **Validation Set (15%):** 982 images
  - **Unseen Test Set (15%):** 984 images
  - **Total Processed Split:** 6,552 images
- **Data Augmentation:** Random resized cropping ($224 \times 224$), horizontal flips ($p=0.5$), vertical flips ($p=0.2$), affine rotations ($\pm 15^\circ$), and color jitter (brightness, contrast, saturation $\pm 0.15$).

---

## 5. Methodology & Architecture

### 5.1 Classical Image Processing (CO1 & CO2)
- **Grayscale Conversion:** Intensity reduction for baseline luminance computation.
- **HSV Color Space Transformation:** Disentangles chromaticity from brightness, crucial for leaf symptom isolation under variable ambient lighting.
- **CLAHE Enhancement:** Applied to the Luminance ($L^*$) channel of the CIELAB color space (`clipLimit=2.5`, tile size $8 \times 8$). Prevents saturation while dramatically amplifying fine leaf vein and lesion boundaries.
- **Edge Detection:**
  - **Sobel Operator:** Directional kernels convolve gradients $G_x$ and $G_y$; combined magnitude: $M(x, y) = 0.5|G_x| + 0.5|G_y|$.
  - **Canny Edge Detector:** Incorporates Gaussian pre-smoothing, gradient computation, non-maximum suppression (NMS), and dual-threshold hysteresis.
- **ORB Feature Extraction:** Detects oriented FAST corner keypoints with BRIEF descriptors, providing rotational and scale invariance.

### 5.2 Lesion Segmentation & Severity Metric (CO3)
- **Leaf Canopy Isolation:** A dual-mask approach combining Otsu grayscale thresholding with saturation thresholding ($S > 25$), followed by morphological opening (radius 3) and closing (radius 7), and connected component labeling to retain only the primary leaf blade.
- **Lesion Isolation:** Multi-interval HSV thresholding identifying brown necrotic tissue ($H \in [0, 32] \cup [160, 180]$), yellow chlorosis, and dark fungal rot ($V < 65$).
- **Quantification Formula:**
  $$\text{Severity (\%)} = \left( \frac{\sum_{(x,y)} \mathbb{I}(\text{Lesion Mask}(x,y) = 255)}{\sum_{(x,y)} \mathbb{I}(\text{Leaf Mask}(x,y) = 255)} \right) \times 100$$
- **Health Categories:**
  - $< 1.0\%$: Healthy / Negligible
  - $1.0\% - 10.0\%$: Mild Infection
  - $10.0\% - 25.0\%$: Moderate Infection
  - $> 25.0\%$: Severe Infection

### 5.3 Deep Learning Classification (CO3 & CO4)
- **Model 1: Custom Baseline CNN (Practical 7):**
  - 4 Sequential Convolutional blocks: `[Conv2D(3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)]` scaling channels: $3 \to 32 \to 64 \to 128 \to 256$.
  - Adaptive Average Pooling $(4 \times 4)$ followed by Flatten, Dense(512), Dropout(0.4), and Dense(33).
- **Model 2: MobileNetV2 (Practical 8):**
  - Inverted residual blocks with depthwise separable convolutions.
  - Initialized with ImageNet pre-trained weights; custom classification head with Dropout(0.3), Linear(1280, 512), ReLU, Dropout(0.2), Linear(512, 33).
  - Optimizer: AdamW (learning rate $3 \times 10^{-4}$, weight decay $1 \times 10^{-4}$) with Cosine Annealing learning rate schedule.

### 5.4 Model Interpretability via Grad-CAM (CO5)
- Backpropagates target class output scores $Y^c$ with respect to the activations $A^k$ of the final feature layer (`model.features[-1]`):
  $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial Y^c}{\partial A_{i,j}^k}$$
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- The resulting activation map is normalized to $[0, 1]$, color-mapped with `JET`, and overlaid with $\alpha=0.45$ transparency onto the input leaf.

---

## 6. Experimental Results & Discussion

### 6.1 Quantitative Model Comparison
Both models were evaluated on the 984 unseen test images:

| Architecture | Parameters | Training Time | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Custom CNN** | ~2.5M | 20.2 mins | **73.98%** | 74.89% | 74.04% | 72.58% |
| **MobileNetV2 (Transfer)** | ~2.8M | 28.5 mins | **96.95%** | **97.02%** | **96.97%** | **96.96%** |

MobileNetV2 achieved a **+22.97% higher test accuracy** than the baseline CNN, proving the superior representational power of ImageNet pre-trained feature hierarchies on complex foliar patterns.

### 6.2 Segmentation & Severity Quantification Results
Evaluated on sample leaves:
- **Tomato (Early Blight):** 51.78% leaf area affected $\to$ **Severe Infection**
- **Corn (Common Rust):** 46.82% leaf area affected $\to$ **Severe Infection**
- **Grape (Black Rot):** 20.07% leaf area affected $\to$ **Moderate Infection**
- **Apple (Cedar Rust):** 9.69% leaf area affected $\to$ **Mild Infection**
- **Tomato (Healthy):** 0.08% leaf area affected $\to$ **Healthy Reference**

---

## 7. System Implementation & Apple Glass UI
- **Local Server (`server.py`):** Standalone zero-dependency Python server running on `http://127.0.0.1:8000`.
- **Glassmorphic Styling (`static/css/glass.css`):** Built using modern CSS `backdrop-filter: blur(26px) saturate(190%)`, translucent card borders, Apple SF Pro typography, and ambient mesh gradient lighting.
- **Interactive Multi-View Tabs:**
  - Tab 1: AI Diagnosis & Grad-CAM Attention Heatmap
  - Tab 2: Lesion Segmentation & Severity Overlay
  - Tab 3: Classical Computer Vision Filters (CLAHE, Sobel, Canny, ORB)
- **Sample Leaf Explorer:** Interactive species filter pills supporting all 9 plant crops for immediate demonstration.

---

## 8. Limitations & In-the-Wild Challenges
1. **Background Artifacts:** Models trained on laboratory-curated datasets (PlantVillage) perform best on single leaves against uniform backdrops. Leaves photographed in outdoor soil beds with overlapping vegetation may experience reduced segmentation accuracy.
2. **Confidence Verification:** To mitigate false positives, the system implements an out-of-distribution confidence threshold ($< 50\%$). If a user uploads an arbitrary non-leaf object, the system flags an advisory warning rather than making a high-confidence erroneous diagnosis.

---

## 9. Conclusion & Future Scope
This project successfully demonstrated all five Computer Vision Course Outcomes (CO1 to CO5) within an integrated, working system:
- Demonstrated classical color conversions and spatial filters (Unit I, II).
- Implemented morphological lesion segmentation and mathematical severity metrics (Unit III).
- Compared a baseline CNN with fine-tuned MobileNetV2 reaching 96.95% accuracy (Unit IV).
- Visualized decision interpretability through Grad-CAM (Unit V).
- Packaged the entire workflow into a responsive Apple-style Glass local web app.

**Future Scope:**
- Integration of Vision Transformers (ViT) for attention-map benchmarking.
- Quantization to ONNX / TensorFlow Lite for real-time offline deployment on mobile Android/iOS farm scouting devices.

