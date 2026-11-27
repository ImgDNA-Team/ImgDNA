"""Barrier 2a: AI / deepfake detection (Image DNA).

Usage:
    from ai_detector import DeepfakeDetector

    detector = DeepfakeDetector("best.pth", "thresholds.json")   # load once
    result = detector.predict("photo.jpg", with_heatmap=True)    # never raises

Command line:
    python ai_detector.py photo.jpg --weights best.pth --thresholds thresholds.json

Notes:
    * `score` (0..1) is an UNCALIBRATED manipulation score, not a probability.
      Decisions should use `label`, which is derived from validated thresholds.
    * `label == "real"` means no manipulation evidence was found. It does not
      prove the image is authentic.
    * The Grad-CAM heat-map is only returned when `label != "real"`.
"""

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import efficientnet_b4

__all__ = ["DeepfakeDetector", "MODEL_VERSION", "load_image"]

MODEL_VERSION = "efficientnet_b4_celebdf_fullframe_v1"
IMAGE_SIZE = 380
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ImageInput = Union[str, Path, bytes, Image.Image]


def load_image(image: ImageInput) -> Image.Image:
    """Open an image, apply its EXIF orientation, and convert it to RGB."""
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(image))
    else:
        pil = Image.open(image)

    try:
        pil = ImageOps.exif_transpose(pil)
    except Exception:
        pass  # unreadable EXIF must not reject an otherwise valid image
    return pil.convert("RGB")


def build_model() -> nn.Module:
    """EfficientNet-B4 with one output logit; sigmoid(logit) is the uncalibrated fake score."""
    model = efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


def build_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Evaluation transform: the whole image is kept (no face crop, aspect ratio not preserved)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def grad_cam(model: nn.Module, input_tensor: torch.Tensor, target_layer: nn.Module) -> np.ndarray:
    """Grad-CAM heat-map (values 0..1) showing what pushed the score towards 'fake'."""
    activations, gradients = [], []

    def forward_hook(module, inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            model(input_tensor).squeeze().backward()
    finally:
        handle.remove()

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def jet_colormap(values: np.ndarray) -> np.ndarray:
    """Jet-like colormap (0..1 -> uint8 RGB) without needing matplotlib."""
    r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat = Image.fromarray(jet_colormap(cam)).resize(image.size)
    return Image.blend(image, heat, alpha)


def encode_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class DeepfakeDetector:
    """Inference interface for Barrier 2a. Create once, call `predict` per image."""

    def __init__(self, weights_path, thresholds_path, device=None, image_size: int = IMAGE_SIZE):
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.image_size = image_size
        self.transform = build_transform(image_size)

        self.model = build_model()
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device).eval()

        with open(thresholds_path) as f:
            thresholds = json.load(f)
        self.low, self.high = float(thresholds["low"]), float(thresholds["high"])
        if self.low > self.high:
            raise ValueError("`low` threshold must not be greater than `high`.")

    def _label(self, score: float) -> str:
        if score >= self.high:
            return "fake"
        return "suspicious" if score >= self.low else "real"

    def _explain_pil(self, image: Image.Image) -> Image.Image:
        resized = image.resize((self.image_size, self.image_size))
        tensor = self.transform(resized).unsqueeze(0).to(self.device)
        cam = grad_cam(self.model, tensor, self.model.features[-1])
        return overlay_heatmap(resized, cam)

    def explain(self, image: ImageInput) -> Image.Image:
        """Grad-CAM overlay on the resized input image (meaningful for non-real images)."""
        return self._explain_pil(load_image(image))

    def predict(self, image: ImageInput, with_heatmap: bool = False) -> dict:
        """Return the barrier result. Failures are reported in `error`, never raised."""
        result = {
            "barrier": "ai_deepfake",
            "model_version": MODEL_VERSION,
            "score": None,
            "label": None,
            "thresholds": {"low": self.low, "high": self.high},
            "heatmap_png_base64": None,
            "error": None,
        }
        try:
            pil_image = load_image(image)
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                score = torch.sigmoid(self.model(tensor)).item()
            result["score"], result["label"] = score, self._label(score)
            if with_heatmap and result["label"] != "real":
                result["heatmap_png_base64"] = encode_png_base64(self._explain_pil(pil_image))
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI/deepfake detector on one image.")
    parser.add_argument("image")
    parser.add_argument("--weights", default="best.pth")
    parser.add_argument("--thresholds", default="thresholds.json")
    parser.add_argument("--heatmap", help="Optional path to save the Grad-CAM overlay (PNG).")
    args = parser.parse_args()

    detector = DeepfakeDetector(args.weights, args.thresholds)
    result = detector.predict(args.image, with_heatmap=bool(args.heatmap))
    heatmap = result.pop("heatmap_png_base64")
    print(json.dumps(result, indent=2))

    if args.heatmap:
        if heatmap:
            Path(args.heatmap).write_bytes(base64.b64decode(heatmap))
            print("Heat-map saved to", args.heatmap)
        else:
            print("No heat-map: the image is labelled 'real' or an error occurred.")


if __name__ == "__main__":
    _main()
