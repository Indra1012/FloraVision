"""Model Interpretability & Explainability Module via Grad-CAM (CO5, Unit V, Practical 9 & 11).
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize
the spatial regions of a leaf that the deep learning model uses to make its diagnosis.
"""

import os
import sys
import json
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
import torch
import torch.nn.functional as F
from torchvision import transforms

from models import build_mobilenet_v2
from segmentation import run_segmentation_pipeline

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for Convolutional Neural Networks."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class_idx=None):
        """Compute Grad-CAM heatmap for a given input tensor and target class."""
        self.model.eval()

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class_idx]

        # Zero gradients and backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Global average pooling of gradients: weights alpha_k
        # gradients shape: [1, C, H, W]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to focus on features having positive influence on the target class
        cam = F.relu(cam)

        # Normalize between 0 and 1
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class_idx, probs[0, target_class_idx].item(), logits


def overlay_heatmap_on_image(img_rgb, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """Overlay resized color heatmap onto original RGB image."""
    h, w = img_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to 8-bit image and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose onto original leaf
    blended = cv2.addWeighted(heatmap_colored, alpha, img_rgb, 1 - alpha, 0)
    return blended, heatmap_resized


def load_model_and_labels(device="cpu"):
    """Load trained MobileNetV2 and class index dictionary."""
    weights_path = MODELS_DIR / "mobilenetv2_best.pth"
    class_map_path = MODELS_DIR / "class_indices.json"

    if not class_map_path.exists():
        raise FileNotFoundError(f"Class mapping not found at: {class_map_path}")

    with open(class_map_path) as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[+] Loaded trained model weights from: {weights_path}")
    else:
        print(f"[!] Warning: Trained weights not found at {weights_path}. Using initialized model for testing.")

    model.to(device)
    model.eval()
    return model, idx_to_class


def explain_leaf_prediction(image_path_or_array, model, idx_to_class, device="cpu"):
    """Full diagnostic explainability pipeline for any leaf image."""
    if isinstance(image_path_or_array, (str, Path)):
        img_bgr = cv2.imread(str(image_path_or_array))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_path_or_array.copy()

    img_resized = cv2.resize(img_rgb, (224, 224))

    # PyTorch Preprocessing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    input_tensor = transform(img_resized).unsqueeze(0).to(device)

    # Attach Grad-CAM to the final convolutional layer of MobileNetV2 features
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    heatmap, pred_idx, confidence, logits = grad_cam.generate_heatmap(input_tensor)
    pred_class_raw = idx_to_class[pred_idx]
    pred_class_clean = pred_class_raw.replace("___", " - ").replace("_", " ")

    # Top-3 predictions
    probs = F.softmax(logits, dim=1)[0]
    top3_vals, top3_indices = torch.topk(probs, 3)
    top3 = [
        {"class": idx_to_class[i.item()].replace("___", " - ").replace("_", " "), "confidence": round(v.item() * 100, 2)}
        for v, i in zip(top3_vals, top3_indices)
    ]

    overlay, _ = overlay_heatmap_on_image(img_resized, heatmap)

    # Also compute classical segmentation for complete diagnostic view
    seg_results = run_segmentation_pipeline(img_resized)

    return {
        "original_img": img_resized,
        "pred_class": pred_class_clean,
        "raw_class": pred_class_raw,
        "confidence_pct": round(confidence * 100, 2),
        "top3": top3,
        "heatmap": heatmap,
        "gradcam_overlay": overlay,
        "segmentation": seg_results,
    }


def generate_explainability_figure(image_path, model, idx_to_class, output_filename, display_name="Plant Leaf"):
    """Generate and save an academic explainability figure (CO5)."""
    results = explain_leaf_prediction(image_path, model, idx_to_class)
    seg = results["segmentation"]
    metrics = seg["metrics"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"AI Explainability & Lesion Verification (CO5) — {display_name}\n"
        f"Model Prediction: {results['pred_class']} ({results['confidence_pct']}% Confidence) | Lesion Severity: {metrics['severity_pct']}%",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )

    panels = [
        (results["original_img"], "1. Input Leaf Image", None),
        (seg["overlay"], f"2. Lesion Segmentation ({metrics['severity_pct']}% affected)", None),
        (results["heatmap"], "3. Grad-CAM Activation Heatmap", "jet"),
        (results["gradcam_overlay"], f"4. Model Attention Overlay ({results['confidence_pct']}%)", None),
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
    print(f"[+] Explainability figure saved: {out_path}")
    return out_path


def main():
    device = torch.device("cpu")
    model, idx_to_class = load_model_and_labels(device=device)

    demo_leaves = [
        ("Tomato___Early_blight_sample.jpg", "gradcam_tomato_early_blight.png", "Tomato (Early Blight)"),
        ("Apple___Cedar_apple_rust_sample.jpg", "gradcam_apple_cedar_rust.png", "Apple (Cedar Apple Rust)"),
        ("Corn_(maize)___Common_rust__sample.jpg", "gradcam_corn_common_rust.png", "Corn (Common Rust)"),
        ("Grape___Black_rot_sample.jpg", "gradcam_grape_black_rot.png", "Grape (Black Rot)"),
        ("Tomato___healthy_sample.jpg", "gradcam_tomato_healthy.png", "Tomato (Healthy Reference)"),
    ]

    print("\n" + "=" * 65)
    print("🔍 Executing Phase 5: Model Explainability via Grad-CAM (CO5)")
    print("=" * 65)

    for filename, output_name, display_title in demo_leaves:
        sample_path = SAMPLES_DIR / filename
        if sample_path.exists():
            generate_explainability_figure(sample_path, model, idx_to_class, output_name, display_title)
        else:
            print(f"[-] Sample image not found: {sample_path}")

    print("\n✅ Phase 5 Grad-CAM explainability pipeline ready!")


if __name__ == "__main__":
    main()

