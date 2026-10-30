# 🌿 FloraVision: Comprehensive Viva Voce & Practical Record Guide
### CIA Mini Project Practical Evaluation (20 Marks | CO1 to CO5)

---

## 📌 Course Outcomes (CO) Mapping Matrix

| Practical / Component | Syllabus Unit | Course Outcome (CO) | Technical Implementation |
| :--- | :--- | :--- | :--- |
| **Practical 2: Color Space Conversions** | **Unit I** | **CO1:** Understand digital image acquisition and color spaces | Converted RGB leaf images to Grayscale and HSV. Separated the Hue channel (chlorophyll detection) from Saturation and Value to isolate necrotic lesions regardless of ambient lighting variations. |
| **Practical 3: Spatial Filtering & Edges** | **Unit II** | **CO2:** Image enhancement and spatial frequency filtering | Applied Contrast Limited Adaptive Histogram Equalization (**CLAHE**) on the Luminance channel to enhance subtle lesion textures. Applied Bilateral filtering to preserve leaf edges while eliminating sensor noise. Computed **Sobel** gradient magnitudes ($G_x$, $G_y$) and **Canny** edge detection with hysteresis thresholding. |
| **Practical 4: Feature Extraction** | **Unit II** | **CO2:** Invariant feature descriptors | Extracted **ORB (Oriented FAST and Rotated BRIEF)** keypoints to capture rotational- and scale-invariant corners across healthy vs infected plant tissue. |
| **Practical 5: Image Segmentation** | **Unit III** | **CO3:** Morphological operations and segmentation | Segmented the leaf canopy from background using Otsu thresholding and saturation masking. Isolated disease spots using HSV necrosis thresholding. Formulated a mathematical metric: $\text{Severity (\%)} = \frac{\text{Lesion Pixels}}{\text{Leaf Pixels}} \times 100$. |
| **Practical 7: Convolutional Networks** | **Unit IV** | **CO4:** CNN architecture & feature hierarchies | Designed a custom 4-block Convolutional Neural Network (Conv2D $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool) to demonstrate understanding of spatial filters, receptive fields, and downsampling. |
| **Practical 8: Transfer Learning** | **Unit IV** | **CO4:** Pretrained models & fine-tuning | Implemented transfer learning using **MobileNetV2** (ImageNet pretrained weights) fine-tuned on 33 multi-crop classes across 9 plant species. Evaluated via confusion matrices, macro precision, recall, and F1-scores. |
| **Practical 9 & 11: Explainability** | **Unit V** | **CO5:** Advanced CV & Interpretability | Implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)** hooking into the final convolutional layer of MobileNetV2. Generated heatmaps proving the model fixates on diseased lesions rather than background artifacts. |

---

## 🎓 High-Probability Viva Voce Questions & Model Answers

### Q1: Why did you convert RGB images to the HSV color space for leaf disease analysis?
> **Answer:** In the RGB color space, chromaticity (color) and luminance (brightness) are deeply coupled across all three channels. Natural leaf images suffer from uneven sunlight, shadows, and camera exposure. The **HSV (Hue, Saturation, Value)** color space decouples chromatic information (**Hue**) from illumination (**Value**). Healthy green plant tissue occupies Hue values between $35^\circ$ and $85^\circ$, whereas diseased necrotic lesions appear as yellow chlorosis ($25^\circ-38^\circ$) or brown/orange necrosis ($0^\circ-30^\circ$ and $160^\circ-180^\circ$). This allows robust segmentation regardless of shadows.

### Q2: Why use CLAHE instead of standard Global Histogram Equalization?
> **Answer:** Standard Global Histogram Equalization computes a single cumulative distribution function (CDF) for the entire image. In plant images with bright backgrounds or dark leaf regions, global equalization over-amplifies background noise and washes out subtle lesion details. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** operates on local tiles (e.g., $8 \times 8$ grid) and clips the local histogram at a predefined contrast threshold (`clipLimit=2.5`), preventing noise over-amplification while dramatically enhancing fine leaf veins and lesion margins.

### Q3: What is the fundamental difference between the Sobel operator and Canny edge detector?
> **Answer:**
> - **Sobel Operator:** Computes first-order spatial derivatives by convolving the image with two $3 \times 3$ directional kernels ($G_x$ for vertical edges, $G_y$ for horizontal edges). It yields gradient magnitude and direction, resulting in thick, continuous gradient responses.
> - **Canny Edge Detector:** A multi-stage algorithm that performs:
>   1. Gaussian smoothing to remove noise.
>   2. Sobel gradient magnitude and orientation calculation.
>   3. **Non-maximum suppression (NMS)** to thin thick edges down to 1-pixel wide ridges.
>   4. **Hysteresis thresholding** with low and high thresholds to retain weak edges only if connected to strong edges, suppressing noise.

### Q4: Why is MobileNetV2 particularly well-suited for this project?
> **Answer:** MobileNetV2 uses **depthwise separable convolutions** and **inverted residual blocks with linear bottlenecks**:
> 1. Standard convolutions compute spatial filtering and channel mixing simultaneously ($D_K \cdot D_K \cdot M \cdot N$). Depthwise separable convolutions split this into a depthwise spatial convolution ($D_K \cdot D_K \cdot M$) followed by a $1 \times 1$ pointwise convolution ($M \cdot N$), reducing computational operations by roughly **8 to 9 times**.
> 2. With only ~2.2 million parameters (compared to ResNet-50's ~25M or VGG-16's ~138M), MobileNetV2 runs in real-time on standard laptop CPUs without requiring dedicated GPUs.

### Q5: How does Grad-CAM work mathematically?
> **Answer:** Grad-CAM computes the importance of each feature map in the last convolutional layer ($A$) for a predicted target class score ($Y^c$):
> 1. Compute gradients of class score $Y^c$ with respect to feature activation maps $A^k$: $\frac{\partial Y^c}{\partial A_{i,j}^k}$.
> 2. Perform **Global Average Pooling** over spatial dimensions ($i, j$) to calculate neuron importance weights $\alpha_k^c$:
>    $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$
> 3. Take a weighted sum of activation maps and apply **ReLU** to isolate features with a positive influence on the disease class:
>    $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
> 4. Resize and overlay the resulting heatmap onto the original image.

### Q6: How does the system handle corrupt, noisy, or non-leaf uploaded images?
> **Answer:** We implemented a two-tier validation pipeline (Phase 6):
> 1. **Structural Sanitization:** Strips alpha transparency channels, handles grayscale inputs, and normalizes image channels to standard ImageNet mean and variance.
> 2. **Confidence Thresholding:** If the model's top softmax probability falls below the confidence threshold ($< 50\%$), the app displays an out-of-distribution advisory warning informing the user that the image is ambiguous or may not be a recognized leaf.

---

## 📊 Summary of Architectural Pipeline

```text
[Input Image (JPG/PNG/WEBP)]
         │
         ├──► Classical Processing (RGB->HSV, CLAHE, Bilateral, Sobel, Canny, ORB) [CO1, CO2]
         │
         ├──► Segmentation Engine (Leaf Mask + Lesion Isolation + Surface % Calculation) [CO3]
         │
         ├──► MobileNetV2 Classifier (33 Multi-Crop Disease Classes) [CO4]
         │
         ├──► Grad-CAM Explainability (Gradient Backpropagation to Final Conv Layer) [CO5]
         │
         └──► Apple-Style Glassmorphism Web UI (Interactive Diagnostic Suite)
```

