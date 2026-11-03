# Barrier 2a: AI / Deepfake Detection

Part of **Image DNA**. This module takes one image and returns a manipulation score, a label (`real` / `suspicious` / `fake`) and an optional Grad-CAM heat-map. The fusion engine combines this output with the watermark barrier.

## Model

- EfficientNet-B4, ImageNet weights, one output logit. `score = sigmoid(logit)` = probability the image is manipulated.
- Input: the whole image resized to 380x380 (no face crop).
- Trained on Celeb-DF v2 frames (5 per video). Train/validation split by video; test set is the official Celeb-DF test list (518 videos, 2,590 frames).
- AdamW (lr 1e-4), FP32, batch size 8, class-balanced sampler (12,000 samples per epoch), early stopping on validation AUC. Best epoch: 5 (validation AUC 0.9704).

## Results (test set, frame level, fake = positive)

| Threshold | AUC | F1 | Precision | Recall | FPR |
|---|---|---|---|---|---|
| 0.5 | 0.9735 | 0.9401 | 0.9379 | 0.9424 | 11.9% |
| 0.997 (FPR <= 5% on validation) | 0.9735 | 0.8551 | 0.9832 | 0.7565 | 2.5% |

Verdict bands (thresholds chosen on the validation set: low = 0.799 for recall >= 95%, high = 0.997 for FPR <= 5%):

| True class | real (< 0.799) | suspicious | fake (>= 0.997) |
|---|---|---|---|
| Real (890) | 807 | 61 | 22 |
| Fake (1,700) | 128 | 286 | 1,286 |

A `fake` verdict is 98.3% precise. A `real` label only means no manipulation evidence was found, it does not prove the image is authentic. The fusion engine should report it as Unverified unless the watermark barrier confirms the image.

Grad-CAM maps on fake test images concentrate on the face region, not on the background or overlaid text.

## Files

| File | Purpose |
|---|---|
| `ai_detector.py` | `DeepfakeDetector` class (inference interface) |
| `best.pth` | Model weights (about 70 MB, keep out of git: use Git LFS or a GitHub Release) |
| `thresholds.json` | `low` and `high` thresholds |
| `model_meta.json` | Model version, dataset, test metrics |
| `training_history.csv` | Per-epoch training and validation metrics |

## Usage

```python
from pathlib import Path
from ai_detector import DeepfakeDetector

detector = DeepfakeDetector(Path("best.pth"), Path("thresholds.json"))
result = detector.predict("photo.jpg", with_heatmap=True)
print(result["label"], result["score"])
```

Output (`predict` never raises: failures are returned in `error`, so the pipeline can degrade gracefully):

```json
{
  "barrier": "ai_deepfake",
  "model_version": "efficientnet_b4_celebdf_fullframe_v1",
  "score": 0.87,
  "label": "suspicious",
  "thresholds": {"low": 0.799, "high": 0.997},
  "heatmap_png_base64": null,
  "error": null
}
```

`detector.explain(image)` returns the Grad-CAM overlay as a PIL image.

Requirements: see `requirements.txt` (`torch`, `torchvision`, `pillow`, `numpy`).

## Limitations

- Training and testing both use Celeb-DF v2, and the same identities appear in train, validation and test. Results are an upper bound on real-world performance.
- The data covers celebrity face-swap videos only. Performance on ID cards, certificates, non-face edits and fully AI-generated images has not been tested.
- The test set has only 178 real videos, so the FPR estimate has a wide margin of error.
- Cross-dataset evaluation and inference-time measurement are still pending.
- Scores are probabilistic. They are decision support, not proof of forgery.
