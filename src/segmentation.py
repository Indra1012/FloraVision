"""Image Segmentation & Disease Severity Quantification Module (CO3, Unit III, Practical 5).
Performs:
1. Leaf-from-background isolation (Color thresholding + Otsu + Morphological ops)
2. Diseased lesion spot extraction (HSV necrosis/chlorosis thresholding)
3. Quantitative leaf disease severity metric calculation (% surface area affected)
4. Visual contour & heat overlay for reporting and the Apple-style Glass UI
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Safe stdout handling for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")

import matplotlib.pyplot as plt

REPORTS_DIR = PROJECT_ROOT / "reports"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


def load_image(image_path_or_array, target_size=(256, 256)):
    """Load image from path or validate numpy array, converting to RGB."""
    if isinstance(image_path_or_array, (str, Path)):
        img = cv2.imread(str(image_path_or_array))
        if img is None:
            raise ValueError(f"Could not load image from: {image_path_or_array}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path_or_array, np.ndarray):
        img = image_path_or_array.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise TypeError("Input must be a file path or numpy array.")

    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img


def segment_leaf(img_rgb):
    """Isolate the leaf from the background using combined color space thresholding and morphology."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Plant tissue covers green foliage, yellow chlorosis, brown necrosis, and dark rot
    mask_plant = cv2.inRange(hsv, (18, 25, 20), (105, 255, 255))
    mask_brown1 = cv2.inRange(hsv, (0, 25, 20), (25, 255, 220))
    mask_brown2 = cv2.inRange(hsv, (160, 25, 20), (180, 255, 220))
    mask_dark = cv2.inRange(hsv, (0, 0, 15), (180, 255, 75))

    combined = cv2.bitwise_or(mask_plant, mask_brown1)
    combined = cv2.bitwise_or(combined, mask_brown2)
    combined = cv2.bitwise_or(combined, mask_dark)

    # Morphological closing to seal internal leaf holes and veins, then opening to eliminate noise
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open, iterations=1)

    # Find leaf contours and fill
    total_area = img_rgb.shape[0] * img_rgb.shape[1]
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_mask = np.zeros_like(cleaned)
    if cnts:
        for c in cnts:
            if cv2.contourArea(c) > 0.03 * total_area:
                cv2.drawContours(leaf_mask, [c], -1, 255, -1)

    # Fallback to Otsu if thresholding found very little (< 5% of image)
    if np.mean(leaf_mask == 255) < 0.05:
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts_otsu, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_otsu:
            c_max = max(cnts_otsu, key=cv2.contourArea)
            cv2.drawContours(leaf_mask, [c_max], -1, 255, -1)

    # Apply mask to image to create the actual segmented leaf blade
    segmented_leaf = cv2.bitwise_and(img_rgb, img_rgb, mask=leaf_mask)
    return leaf_mask, segmented_leaf


def segment_lesions(img_rgb, leaf_mask):
    """Isolate diseased necrosis, chlorosis, and rust spots within the segmented leaf mask."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Healthy plant tissue has green hues (H in [35, 88]) with moderate saturation
    # Diseased tissue shows up as:
    # 1. Brown / Red / Orange necrosis: H in [0, 30] or [160, 180]
    # 2. Yellow chlorosis: H in [25, 38] with distinct saturation
    # 3. Dark fungal lesions / rot: Very low Value (V < 60) inside the leaf
    lower_brown1 = np.array([0, 30, 20])
    upper_brown1 = np.array([32, 255, 220])
    mask_brown1 = cv2.inRange(hsv, lower_brown1, upper_brown1)

    lower_brown2 = np.array([160, 30, 20])
    upper_brown2 = np.array([180, 255, 220])
    mask_brown2 = cv2.inRange(hsv, lower_brown2, upper_brown2)

    # Dark rot spots
    mask_dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 65]))

    # Combine disease candidates
    raw_disease = cv2.bitwise_or(mask_brown1, mask_brown2)
    raw_disease = cv2.bitwise_or(raw_disease, mask_dark)

    # Strictly constrain to the segmented leaf area only
    disease_within_leaf = cv2.bitwise_and(raw_disease, leaf_mask)

    # Morphological filtering to eliminate isolated single-pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_disease = cv2.morphologyEx(disease_within_leaf, cv2.MORPH_OPEN, kernel, iterations=1)
    clean_disease = cv2.morphologyEx(clean_disease, cv2.MORPH_CLOSE, kernel, iterations=1)

    return clean_disease


def calculate_severity_metrics(leaf_mask, disease_mask):
    """Calculate the quantitative disease severity percentage and health category."""
    leaf_pixels = int(np.count_nonzero(leaf_mask))
    disease_pixels = int(np.count_nonzero(disease_mask))

    if leaf_pixels == 0:
        severity_pct = 0.0
    else:
        severity_pct = (disease_pixels / leaf_pixels) * 100.0

    # Categorize severity
    if severity_pct < 1.0:
        status = "Healthy / Negligible"
        status_color = (46, 125, 50)     # Green
    elif severity_pct < 10.0:
        status = "Mild Infection"
        status_color = (251, 192, 45)    # Yellow
    elif severity_pct < 25.0:
        status = "Moderate Infection"
        status_color = (245, 124, 0)     # Orange
    else:
        status = "Severe Infection"
        status_color = (211, 47, 47)     # Red

    return {
        "leaf_pixels": leaf_pixels,
        "disease_pixels": disease_pixels,
        "severity_pct": round(severity_pct, 2),
        "status": status,
        "status_color": status_color,
    }


def draw_segmentation_overlay(img_rgb, leaf_mask, disease_mask, metrics):
    """Draw vibrant translucent overlays and contours highlighting diseased regions."""
    overlay = img_rgb.copy()

    # Create colored translucent mask for disease (Red/Orange tint)
    disease_colored = np.zeros_like(img_rgb)
    disease_colored[disease_mask > 0] = [230, 40, 40]  # Vibrant Red

    # Blend original and colored mask
    alpha = 0.45
    cv2.addWeighted(disease_colored, alpha, overlay, 1 - alpha, 0, overlay)

    # Draw sharp contours around lesions
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 235, 59), 1)  # Neon Yellow boundary line

    # Draw Leaf perimeter contour (Subtle cyan line)
    leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, leaf_contours, -1, (0, 229, 255), 1)

    return overlay


def run_segmentation_pipeline(image_path_or_array):
    """Unified segmentation pipeline returning masks, metrics, and visual overlays."""
    original = load_image(image_path_or_array)
    leaf_mask, segmented_leaf = segment_leaf(original)
    disease_mask = segment_lesions(original, leaf_mask)
    metrics = calculate_severity_metrics(leaf_mask, disease_mask)
    overlay = draw_segmentation_overlay(original, leaf_mask, disease_mask, metrics)

    return {
        "original": original,
        "leaf_mask": leaf_mask,
        "segmented_leaf": segmented_leaf,
        "disease_mask": disease_mask,
        "overlay": overlay,
        "metrics": metrics,
    }


def generate_segmentation_figure(image_path, output_filename, plant_name="Plant Leaf"):
    """Generate and save an academic multi-panel figure for lab practical 5."""
    results = run_segmentation_pipeline(image_path)
    metrics = results["metrics"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"Leaf & Lesion Segmentation (CO3, Practical 5) — {plant_name}\n"
        f"Affected Area: {metrics['severity_pct']}% | Status: {metrics['status']}",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )

    panels = [
        (results["original"], "1. Original Leaf", None),
        (results["leaf_mask"], f"2. Leaf Blade Mask ({metrics['leaf_pixels']:,} px)", "gray"),
        (results["disease_mask"], f"3. Lesion Spots ({metrics['disease_pixels']:,} px)", "hot"),
        (results["overlay"], f"4. Lesion Overlay ({metrics['severity_pct']}%)", None),
    ]

    for ax, (img_data, title, cmap) in zip(axes, panels):
        if cmap:
            ax.imshow(img_data, cmap=cmap)
        else:
            ax.imshow(img_data)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    plt.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / output_filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Segmentation figure saved: {out_path} ({metrics['severity_pct']}% affected)")
    return out_path


def run_demo_across_sample_plants():
    """Run segmentation pipeline across multiple distinct diseased and healthy leaves."""
    demo_leaves = [
        ("Tomato___Early_blight_sample.jpg", "segmentation_tomato_early_blight.png", "Tomato (Early Blight)"),
        ("Apple___Cedar_apple_rust_sample.jpg", "segmentation_apple_cedar_rust.png", "Apple (Cedar Apple Rust)"),
        ("Corn_(maize)___Common_rust__sample.jpg", "segmentation_corn_common_rust.png", "Corn (Common Rust)"),
        ("Grape___Black_rot_sample.jpg", "segmentation_grape_black_rot.png", "Grape (Black Rot)"),
        ("Tomato___healthy_sample.jpg", "segmentation_tomato_healthy.png", "Tomato (Healthy Reference)"),
    ]

    print("\n" + "=" * 65)
    print("🔬 Executing Phase 3: Image Segmentation & Severity Metric (CO3)")
    print("=" * 65)

    summary_results = []
    for filename, output_name, display_title in demo_leaves:
        sample_path = SAMPLES_DIR / filename
        if sample_path.exists():
            out_path = generate_segmentation_figure(sample_path, output_name, display_title)
            res = run_segmentation_pipeline(sample_path)
            summary_results.append((display_title, res["metrics"]))
        else:
            print(f"[-] Sample image not found: {sample_path}")

    # Generate a comprehensive summary comparison figure
    fig, axes = plt.subplots(len(demo_leaves), 4, figsize=(16, 3.5 * len(demo_leaves)))
    fig.suptitle("Plant Leaf Lesion Segmentation & Severity Metric Evaluation (CO3, Practical 5)",
                 fontsize=16, fontweight="bold", y=0.99)

    for row_idx, (filename, _, display_title) in enumerate(demo_leaves):
        sample_path = SAMPLES_DIR / filename
        res = run_segmentation_pipeline(sample_path)
        m = res["metrics"]

        axes[row_idx, 0].imshow(res["original"])
        axes[row_idx, 0].set_title(f"{display_title}\nOriginal", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(res["segmented_leaf"])
        axes[row_idx, 1].set_title(f"Leaf Isolated\n({m['leaf_pixels']:,} px)", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(res["disease_mask"], cmap="hot")
        axes[row_idx, 2].set_title(f"Lesion Mask\n({m['disease_pixels']:,} px)", fontsize=10, fontweight="bold")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(res["overlay"])
        axes[row_idx, 3].set_title(f"Severity: {m['severity_pct']}%\nStatus: {m['status']}",
                                    fontsize=10, fontweight="bold", color="darkred" if m['severity_pct'] > 1.0 else "green")
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    summary_path = REPORTS_DIR / "segmentation_multi_plant_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[+] Multi-plant segmentation summary chart saved: {summary_path}")
    print("✅ Phase 3 segmentation pipeline successfully completed!")


if __name__ == "__main__":
    run_demo_across_sample_plants()

