# Barrier 2a: AI / Deepfake Detection

Part of **Image DNA**. Takes one image and returns a manipulation score, a label (`real` / `suspicious` / `fake`) and, for non-real labels, an optional Grad-CAM heat-map. The fusion engine combines this output with the watermark barrier.

## How to read the output

- `score` (0 to 1) is an **uncalibrated manipulation score** (sigmoid of the model logit). It is **not a probability**: values like 1e-8 or 0.999 do not mean "certainly real" or "certainly fake". Use it for ranking only.
- **Decisions must use `label`**, which comes from thresholds validated on held-out images.
- `label == "real"` means **no manipulation evidence was found**. It does not prove the image is authentic. The fusion engine should report it as *Unverified*, and *Verified* only when the watermark barrier confirms the image.
- `heatmap_png_base64` is returned only when `label != "real"`. Grad-CAM shows regions that pushed the score towards "fake", so it is not meaningful for real-labelled images.

## Model

- EfficientNet-B4 (ImageNet weights), one output logit.
- Input: the whole image resized to 380x380 (no face crop, aspect ratio not preserved). EXIF orientation is applied and the image is converted to RGB before inference.
- Trained on Celeb-DF v2 frames (5 per video). Train/validation split by video; test set is the official Celeb-DF test list (518 videos, 2,590 frames).
- AdamW (lr 1e-4), FP32, batch size 8, class-balanced sampler (12,000 samples per epoch), early stopping on validation AUC. Best epoch: 5 (validation AUC 0.9704).
- Scope: image-level detection. Every image is scored independently.

## Results (test set, image level, fake = positive)

| Threshold | AUC | F1 | Precision | Recall | FPR |
|---|---|---|---|---|---|
| 0.5 | 0.9735 | 0.9401 | 0.9379 | 0.9424 | 11.9% |
| 0.997 (FPR <= 5% on validation) | 0.9735 | 0.8551 | 0.9832 | 0.7565 | 2.5% |

Confusion counts on the test set (890 real, 1,700 fake images):

| Threshold | TN | FP | FN | TP |
|---|---|---|---|---|
| 0.5 | 784 | 106 | 98 | 1,602 |
| 0.997 | 868 | 22 | 414 | 1,286 |

Verdict bands (thresholds chosen on validation images: low = 0.799 for recall >= 95%, high = 0.997 for FPR <= 5%):

| True class | real (< 0.799) | suspicious | fake (>= 0.997) |
|---|---|---|---|
| Real (890) | 807 | 61 | 22 |
| Fake (1,700) | 128 | 286 | 1,286 |

A `fake` label is 98.3% precise on this test set. 82% of images in the `suspicious` band are fake.

Grad-CAM maps on fake test images concentrate on the face region, not on the background or overlaid text (checked on a handful of images, not a proof).

## Repository layout

```
AI_Detector/
├── README.md
├── ai_detector.py            inference module (DeepfakeDetector)
├── test.py                   smoke test
├── requirements.txt          inference requirements
├── requirements-train.txt    extra packages for the notebook
├── best.pth                  model weights (about 71 MB, below GitHub's 100 MB file limit)
├── thresholds.json           low / high thresholds
├── model_meta.json           model version, dataset, test metrics
├── results/                  training_history.csv, test_predictions.csv
├── images/                   test.jpg, test.png (sample images for the smoke test)
└── Notebook/AI_Detector.ipynb   training, evaluation and export notebook
```

## Usage

```bash
pip install -r requirements.txt
python test.py images/test.jpg                        # smoke test
python ai_detector.py images/test.jpg --heatmap heatmap.png
```

```python
from ai_detector import DeepfakeDetector

detector = DeepfakeDetector("best.pth", "thresholds.json")   # load once, reuse
result = detector.predict("photo.jpg", with_heatmap=True)    # path, bytes or PIL image
```

`predict` never raises: failures are returned in `error`, so the pipeline can degrade gracefully.

```json
{
  "barrier": "ai_deepfake",
  "model_version": "efficientnet_b4_celebdf_fullframe_v1",
  "score": 0.87,
  "label": "suspicious",
  "thresholds": {"low": 0.799, "high": 0.997},
  "heatmap_png_base64": "<PNG, only when label is not real>",
  "error": null
}
```

Integration notes: load the detector once per service process; validate file type and size before calling; if `error` is not null, continue with the other barrier and report that this barrier failed.

## Limitations

- **Narrow training data.** Celeb-DF v2 contains celebrity face-swap videos only. Detection of other manipulation types (edits made in image editors, masks, non-face edits) and of fully AI-generated images is **not demonstrated**. An informal check with a face-mask edit scored about 1e-7 and was labelled `real`, which illustrates this gap.
- **Optimistic test numbers.** Training and test use the same dataset and overlapping identities, so the results above are an upper bound for real-world use. A cross-dataset evaluation has not been run yet.
- **Dataset bias is possible.** Real and fake videos come from different sources and compression pipelines. `AI_Detector.ipynb` section 6 contains per-source AUC, JPEG/resize/blur robustness, bootstrap confidence intervals (resampled by source video) and a latency measurement; add their numbers here after running it.
- **Few real samples.** The test set has 178 real videos, so the FPR estimate has a wide margin of error.
- **Resolution and aspect ratio.** Every image is squashed to 380x380, which may weaken artifacts in high-resolution photos.
- Scores are decision support, not proof of forgery.
