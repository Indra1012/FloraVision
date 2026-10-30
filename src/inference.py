"""Unified Diagnostic & Inference Pipeline (CO1 through CO5).
Accepts any image format (JPG, PNG, WEBP, BMP), validates channel layout,
runs MobileNetV2 classification, Grad-CAM explainability,
classical feature extraction, and lesion severity segmentation.
"""

import os
import sys
import json
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
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

import torch
import torch.nn.functional as F
from torchvision import transforms

from models import build_mobilenet_v2
from classical_processing import process_leaf_pipeline
from segmentation import run_segmentation_pipeline
from explainability import GradCAM, overlay_heatmap_on_image

MODELS_DIR = PROJECT_ROOT / "models"
CONFIDENCE_THRESHOLD = 50.0  # Percentage threshold for out-of-distribution rejection


class FloraVisionPipeline:
    """Master inference engine orchestrating classical CV, segmentation, classification, and Grad-CAM."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = None
        self.idx_to_class = {}
        self.class_to_idx = {}
        self.grad_cam = None

        self._load_model_and_classes()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model_and_classes(self):
        class_map_file = MODELS_DIR / "class_indices.json"
        weights_file = MODELS_DIR / "mobilenetv2_best.pth"

        if class_map_file.exists():
            with open(class_map_file) as f:
                self.class_to_idx = json.load(f)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            num_classes = len(self.idx_to_class)
        else:
            num_classes = 33

        self.model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)

        if weights_file.exists():
            self.model.load_state_dict(torch.load(weights_file, map_location=self.device))
            print(f"[+] Loaded model weights from: {weights_file}")
        else:
            print("[!] Warning: MobileNetV2 trained weights not found yet. Using untrained base.")

        self.model.to(self.device)
        self.model.eval()

        # Initialize Grad-CAM on final feature layer
        self.grad_cam = GradCAM(self.model, self.model.features[-1])

    def preprocess_uploaded_image(self, image_input):
        """Universal image ingestion supporting JPG, PNG with alpha, WEBP, BMP."""
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input)
        elif isinstance(image_input, bytes):
            pil_img = Image.open(BytesIO(image_input))
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise TypeError("Unsupported image input type.")

        # Convert palette or RGBA/Grayscale to standard 3-channel RGB
        rgb_img = pil_img.convert("RGB")
        img_np = np.array(rgb_img)
        return rgb_img, img_np

    def numpy_to_base64(self, img_np, is_bgr=False):
        """Encode numpy image into PNG base64 data URI string for frontend display."""
        if is_bgr:
            img_bgr = img_np
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        success, buffer = cv2.imencode(".png", img_bgr)
        if not success:
            return ""
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def analyze(self, image_input):
        """Run complete end-to-end diagnostic pipeline on an uploaded leaf image."""
        pil_img, img_rgb = self.preprocess_uploaded_image(image_input)
        h, w = img_rgb.shape[:2]

        # 1. Classical Image Processing (CO1 & CO2)
        classical = process_leaf_pipeline(img_rgb)

        # 2. Lesion Segmentation & Severity Metric (CO3)
        segmentation = run_segmentation_pipeline(img_rgb)
        severity_metrics = segmentation["metrics"]

        # 3. Deep Learning Classification (CO4)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 4. Grad-CAM Explainability (CO5)
        heatmap, pred_idx, confidence, logits = self.grad_cam.generate_heatmap(input_tensor)
        confidence_pct = round(confidence * 100, 2)

        probs = F.softmax(logits, dim=1)[0]
        top3_vals, top3_indices = torch.topk(probs, min(3, len(probs)))

        raw_pred_class = self.idx_to_class.get(pred_idx, f"Class_{pred_idx}")
        # Parse plant and condition
        parts = raw_pred_class.split("___")
        plant_name = parts[0].replace("_", " ")
        disease_name = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

        top3 = []
        for val, idx in zip(top3_vals, top3_indices):
            c_name = self.idx_to_class.get(idx.item(), f"Class_{idx.item()}")
            c_parts = c_name.split("___")
            c_plant = c_parts[0].replace("_", " ")
            c_disease = c_parts[1].replace("_", " ") if len(c_parts) > 1 else "Unknown"
            top3.append({
                "label": f"{c_plant}: {c_disease}",
                "confidence": round(val.item() * 100, 2),
            })

        # Out-of-distribution / Low confidence verification (Phase 6)
        is_confident = confidence_pct >= CONFIDENCE_THRESHOLD
        validation_warning = None
        if not is_confident:
            validation_warning = (
                f"Confidence is below threshold ({confidence_pct}% < {CONFIDENCE_THRESHOLD}%). "
                "The image may be ambiguous, low resolution, or not a valid plant leaf."
            )

        # Generate Grad-CAM Visual Overlay
        gradcam_overlay, _ = overlay_heatmap_on_image(cv2.resize(img_rgb, (256, 256)), heatmap)

        # Base64 encodings for UI
        b64_original = self.numpy_to_base64(img_rgb)
        b64_gradcam = self.numpy_to_base64(gradcam_overlay)
        b64_seg_overlay = self.numpy_to_base64(segmentation["overlay"])
        b64_leaf_mask = self.numpy_to_base64(segmentation["segmented_leaf"])
        b64_disease_mask = self.numpy_to_base64(segmentation["disease_mask"])
        b64_clahe = self.numpy_to_base64(classical["clahe"])
        b64_canny = self.numpy_to_base64(classical["canny"])
        b64_sobel = self.numpy_to_base64(classical["sobel_mag"])
        b64_orb = self.numpy_to_base64(classical["orb"])

        return {
            "prediction": {
                "plant": plant_name,
                "disease": disease_name,
                "full_label": f"{plant_name}: {disease_name}",
                "confidence_pct": confidence_pct,
                "is_healthy": "healthy" in disease_name.lower(),
                "top3": top3,
                "is_confident": is_confident,
                "validation_warning": validation_warning,
            },
            "severity": {
                "affected_area_pct": severity_metrics["severity_pct"],
                "health_status": severity_metrics["status"],
                "leaf_pixels": severity_metrics["leaf_pixels"],
                "disease_pixels": severity_metrics["disease_pixels"],
            },
            "classical_cv": {
                "num_orb_keypoints": classical["num_keypoints"],
            },
            "visualizations": {
                "original": b64_original,
                "gradcam_overlay": b64_gradcam,
                "segmentation_overlay": b64_seg_overlay,
                "leaf_mask": b64_leaf_mask,
                "disease_mask": b64_disease_mask,
                "clahe": b64_clahe,
                "canny": b64_canny,
                "sobel": b64_sobel,
                "orb": b64_orb,
            },
        }


# Singleton pipeline instance for reuse
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = FloraVisionPipeline()
    return _pipeline

