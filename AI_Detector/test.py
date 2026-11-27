"""Smoke test for ai_detector.py.  Usage: python test.py <image> [weights] [thresholds]"""

import io
import sys

from PIL import Image

from ai_detector import DeepfakeDetector, load_image


def exif_rotated_jpeg() -> bytes:
    """A 200x100 JPEG whose EXIF orientation says 'rotate 90 degrees'."""
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), (120, 90, 60)).save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def main() -> None:
    image_path = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) > 2 else "best.pth"
    thresholds = sys.argv[3] if len(sys.argv) > 3 else "thresholds.json"

    # EXIF orientation is applied (phone photos)
    assert load_image(exif_rotated_jpeg()).size == (100, 200), "EXIF orientation not applied"

    detector = DeepfakeDetector(weights, thresholds)

    # normal prediction; heat-map only for non-real labels
    result = detector.predict(image_path, with_heatmap=True)
    assert result["error"] is None, result["error"]
    assert 0.0 <= result["score"] <= 1.0
    assert result["label"] in {"real", "suspicious", "fake"}
    assert (result["heatmap_png_base64"] is not None) == (result["label"] != "real")

    # missing file must be reported, not raised
    bad = detector.predict("missing.jpg")
    assert bad["error"] is not None and bad["score"] is None

    # raw bytes (what the API passes) must give the same score as a path
    with open(image_path, "rb") as f:
        from_bytes = detector.predict(f.read())
    assert abs(from_bytes["score"] - result["score"]) < 1e-4

    print("label:", result["label"], "| score:", round(result["score"], 4))
    print("All checks passed")


if __name__ == "__main__":
    main()
