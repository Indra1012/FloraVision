"""Academic Jupyter Notebook Generator for Computer Vision Practical Evaluation (CO1 to CO5).
Generates interactive step-by-step notebooks for lab record submission.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_notebook(filename, title, cells_data):
    nb = {
        "cells": [],
        "metadata": {
            "language_info": {"name": "python", "version": "3.13"},
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    for cell_type, content in cells_data:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + "\n" for line in content.split("\n")]
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)

    out_file = NOTEBOOKS_DIR / filename
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[+] Created academic notebook: {out_file}")


def main():
    # 1. Classical Processing Notebook (CO1 & CO2)
    create_notebook(
        "01_classical_processing.ipynb",
        "Practical 2, 3, 4: Classical Image Processing",
        [
            ("markdown", "# 🌿 Practical 2, 3 & 4: Classical Image Processing (CO1 & CO2)\n**Unit I & II:** Color Space Conversions, CLAHE Enhancement, Smoothing, Edge Detection, and ORB Keypoints."),
            ("code", "import sys\nfrom pathlib import Path\nimport matplotlib.pyplot as plt\n\n# Add src to path\nsys.path.append('../src')\nfrom classical_processing import process_leaf_pipeline, load_image"),
            ("markdown", "### 1. Load Sample Leaf and Execute Classical Pipeline"),
            ("code", "sample_leaf = '../data/samples/Tomato___Early_blight_sample.jpg'\nresults = process_leaf_pipeline(sample_leaf)\nprint(f'Detected ORB keypoints: {results[\"num_keypoints\"]} points')"),
            ("markdown", "### 2. Visualize Multi-Panel Transformations (CO1 & CO2)"),
            ("code", "fig, axes = plt.subplots(2, 4, figsize=(18, 9))\npanels = [\n    (results['original'], 'Original RGB Leaf', None),\n    (results['grayscale'], 'Grayscale (CO1)', 'gray'),\n    (results['hue'], 'HSV: Hue Channel (Chlorophyll)', 'hsv'),\n    (results['saturation'], 'HSV: Saturation (Lesions)', 'inferno'),\n    (results['clahe'], 'CLAHE Enhanced (CO2)', None),\n    (results['sobel_mag'], 'Sobel Gradient Magnitude (CO2)', 'magma'),\n    (results['canny'], 'Canny Edge Map (CO2)', 'gray'),\n    (results['orb'], f'ORB Keypoints: {results[\"num_keypoints\"]} pts (CO2)', None),\n]\nfor ax, (img, title, cmap) in zip(axes.flat, panels):\n    ax.imshow(img, cmap=cmap)\n    ax.set_title(title, fontweight='bold')\n    ax.axis('off')\nplt.tight_layout()\nplt.show()"),
        ]
    )

    # 2. Segmentation Notebook (CO3)
    create_notebook(
        "02_segmentation_analysis.ipynb",
        "Practical 5: Image Segmentation & Severity Metric",
        [
            ("markdown", "# 🔬 Practical 5: Leaf & Lesion Segmentation (CO3)\n**Unit III:** Leaf isolation, HSV necrosis extraction, and quantitative disease severity calculation."),
            ("code", "import sys\nsys.path.append('../src')\nfrom segmentation import run_segmentation_pipeline\nimport matplotlib.pyplot as plt"),
            ("markdown", "### 1. Run Segmentation and Extract Quantitative Metrics"),
            ("code", "sample_leaf = '../data/samples/Corn_(maize)___Common_rust__sample.jpg'\nres = run_segmentation_pipeline(sample_leaf)\nm = res['metrics']\nprint(f'Leaf Area: {m[\"leaf_pixels\"]:,} px | Lesion Area: {m[\"disease_pixels\"]:,} px')\nprint(f'Surface Area Affected: {m[\"severity_pct\"]}% | Condition: {m[\"status\"]}')"),
            ("markdown", "### 2. Visualize Multi-Stage Segmentation Masks"),
            ("code", "fig, axes = plt.subplots(1, 4, figsize=(18, 5))\naxes[0].imshow(res['original'])\naxes[0].set_title('1. Original Leaf', fontweight='bold')\naxes[1].imshow(res['leaf_mask'], cmap='gray')\naxes[1].set_title('2. Leaf Blade Mask', fontweight='bold')\naxes[2].imshow(res['disease_mask'], cmap='hot')\naxes[2].set_title('3. Isolated Lesions', fontweight='bold')\naxes[3].imshow(res['overlay'])\naxes[3].set_title(f'4. Severity Overlay: {m[\"severity_pct\"]}%', fontweight='bold')\nfor ax in axes: ax.axis('off')\nplt.tight_layout()\nplt.show()"),
        ]
    )

    # 3. Explainability Notebook (CO5)
    create_notebook(
        "03_gradcam_explainability.ipynb",
        "Practical 9 & 11: Grad-CAM Explainability",
        [
            ("markdown", "# 🔍 Practical 9 & 11: Deep Learning Diagnosis & Grad-CAM Explainability (CO4 & CO5)\n**Unit V:** Visualizing convolutional feature activation heatmaps to prove model decision interpretability."),
            ("code", "import sys\nimport torch\nsys.path.append('../src')\nfrom explainability import load_model_and_labels, explain_leaf_prediction\nimport matplotlib.pyplot as plt"),
            ("markdown", "### 1. Load Trained MobileNetV2 and Run Diagnostic Inference"),
            ("code", "model, idx_to_class = load_model_and_labels(device='cpu')\nsample_img = '../data/samples/Grape___Black_rot_sample.jpg'\ndiag = explain_leaf_prediction(sample_img, model, idx_to_class)\nprint(f'AI Diagnosis: {diag[\"pred_class\"]} ({diag[\"confidence_pct\"]}% Confidence)')\nfor item in diag['top3']:\n    print(f'  • {item[\"class\"]}: {item[\"confidence\"]}%')"),
            ("markdown", "### 2. Display Side-by-Side Explainability Heatmaps"),
            ("code", "fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))\nax1.imshow(diag['original_img'])\nax1.set_title('Input Leaf', fontweight='bold')\nax2.imshow(diag['heatmap'], cmap='jet')\nax2.set_title('Grad-CAM Heatmap', fontweight='bold')\nax3.imshow(diag['gradcam_overlay'])\nax3.set_title(f'Attention Overlay ({diag[\"confidence_pct\"]}%)', fontweight='bold')\nfor ax in [ax1, ax2, ax3]: ax.axis('off')\nplt.tight_layout()\nplt.show()"),
        ]
    )


if __name__ == "__main__":
    main()

