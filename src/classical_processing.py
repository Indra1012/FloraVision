"""Classical Image Processing Pipeline (CO1 & CO2, Practicals 2, 3, 4).
Covers color space conversions, contrast enhancement (CLAHE), spatial smoothing,
directional Sobel & Canny edge detection, and ORB feature keypoint extraction.
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


def rgb_to_grayscale(img_rgb):
    """Convert RGB to 8-bit Grayscale (CO1, Practical 2)."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def rgb_to_hsv(img_rgb):
    """Convert RGB to HSV color space (CO1, Practical 2)."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)


def apply_clahe(img_rgb, clip_limit=2.5, tile_grid_size=(8, 8)):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on Luminance channel (CO2, Practical 3)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([l_clahe, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def apply_smoothing(img_rgb, method="bilateral"):
    """Apply spatial smoothing filters to reduce sensor noise (CO2, Practical 3)."""
    if method == "gaussian":
        return cv2.GaussianBlur(img_rgb, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(img_rgb, 5)
    elif method == "bilateral":
        # Bilateral filter preserves sharp leaf and lesion edges while smoothing uniform textures
        return cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def detect_edges_sobel(img_rgb):
    """Compute directional Sobel gradients and overall gradient magnitude (CO2, Practical 3)."""
    gray = rgb_to_grayscale(img_rgb)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    abs_x = cv2.convertScaleAbs(grad_x)
    abs_y = cv2.convertScaleAbs(grad_y)
    magnitude = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    return magnitude, abs_x, abs_y


def detect_edges_canny(img_rgb, low_threshold=50, high_threshold=150):
    """Compute Canny edge map with hysteresis thresholding (CO2, Practical 3)."""
    gray = rgb_to_grayscale(img_rgb)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    return cv2.Canny(blurred, low_threshold, high_threshold)


def extract_orb_keypoints(img_rgb, n_features=500):
    """Extract ORB (Oriented FAST and Rotated BRIEF) keypoints and draw descriptors (CO2, Practical 4)."""
    gray = rgb_to_grayscale(img_rgb)
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw keypoints on image (rich green circles with orientation vectors)
    out_img = cv2.drawKeypoints(
        img_rgb,
        keypoints,
        None,
        color=(0, 255, 120),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return out_img, keypoints, descriptors


def process_leaf_pipeline(image_path_or_array):
    """Unified classical processing pipeline returning all intermediate visual representations."""
    original = load_image(image_path_or_array)
    gray = rgb_to_grayscale(original)
    hsv = rgb_to_hsv(original)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    smoothed = apply_smoothing(original, method="bilateral")
    clahe_enhanced = apply_clahe(original)
    sobel_mag, sobel_x, sobel_y = detect_edges_sobel(original)
    canny = detect_edges_canny(original)
    orb_img, keypoints, _ = extract_orb_keypoints(original)

    return {
        "original": original,
        "grayscale": gray,
        "hsv": hsv,
        "hue": h_channel,
        "saturation": s_channel,
        "value": v_channel,
        "smoothed": smoothed,
        "clahe": clahe_enhanced,
        "sobel_mag": sobel_mag,
        "canny": canny,
        "orb": orb_img,
        "num_keypoints": len(keypoints),
    }


def generate_comparison_figure(image_path, output_filename, plant_name="Plant Leaf"):
    """Generate and save an academic multi-panel figure for lab reports."""
    results = process_leaf_pipeline(image_path)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        f"Classical Computer Vision Processing (CO1 & CO2) — {plant_name}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    panels = [
        (results["original"], "1. Original RGB Leaf", None),
        (results["grayscale"], "2. Grayscale (CO1)", "gray"),
        (results["hue"], "3. HSV: Hue Channel (Chlorophyll)", "hsv"),
        (results["saturation"], "4. HSV: Saturation (Lesions)", "inferno"),
        (results["clahe"], "5. CLAHE Enhanced (CO2)", None),
        (results["sobel_mag"], "6. Sobel Gradient Magnitude (CO2)", "magma"),
        (results["canny"], "7. Canny Edge Map (CO2)", "gray"),
        (results["orb"], f"8. ORB Keypoints: {results['num_keypoints']} pts (CO2)", None),
    ]

    for ax, (img_data, title, cmap) in zip(axes.flat, panels):
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
    print(f"[+] Classical comparison figure saved: {out_path}")
    return out_path


def run_demo_across_sample_plants():
    """Run classical pipeline across multiple distinct plant samples and save figures."""
    demo_leaves = [
        ("Tomato___Early_blight_sample.jpg", "classical_tomato_early_blight.png", "Tomato (Early Blight)"),
        ("Apple___Cedar_apple_rust_sample.jpg", "classical_apple_cedar_rust.png", "Apple (Cedar Apple Rust)"),
        ("Corn_(maize)___Common_rust__sample.jpg", "classical_corn_common_rust.png", "Corn (Common Rust)"),
        ("Grape___Black_rot_sample.jpg", "classical_grape_black_rot.png", "Grape (Black Rot)"),
        ("Tomato___healthy_sample.jpg", "classical_tomato_healthy.png", "Tomato (Healthy Reference)"),
    ]

    print("\n" + "=" * 65)
    print("🔬 Executing Phase 2: Classical Image Processing Pipeline (CO1 & CO2)")
    print("=" * 65)

    for filename, output_name, display_title in demo_leaves:
        sample_path = SAMPLES_DIR / filename
        if sample_path.exists():
            generate_comparison_figure(sample_path, output_name, display_title)
        else:
            print(f"[-] Sample image not found: {sample_path}")

    print("\n✅ Phase 2 classical pipeline successfully executed across multiple plant crops!")


if __name__ == "__main__":
    run_demo_across_sample_plants()

