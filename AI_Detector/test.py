import sys

from ai_detector import DeepfakeDetector

image_path = sys.argv[1]
detector = DeepfakeDetector("best.pth", "thresholds.json")

# 1) normal prediction from a path, with heat-map
result = detector.predict(image_path, with_heatmap=True)
assert result["error"] is None, result["error"]
assert 0.0 <= result["score"] <= 1.0
assert result["label"] in {"real", "suspicious", "fake"}
assert result["heatmap_png_base64"]

# 2) missing file must not raise, it must report the error
bad = detector.predict("missing.jpg")
assert bad["error"] is not None and bad["score"] is None

# 3) raw bytes (what the API will pass) must give the same score
with open(image_path, "rb") as f:
    from_bytes = detector.predict(f.read())
assert abs(from_bytes["score"] - result["score"]) < 1e-4

print("label:", result["label"], "| score:", round(result["score"], 4))
print("All checks passed")