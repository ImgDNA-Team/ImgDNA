import cv2
import numpy as np
from app.watermark_utils import InvisibleWatermark


def calculate_psnr(original_path: str, watermarked_path: str) -> float:
    orig = cv2.imread(original_path)
    wm_img = cv2.imread(watermarked_path)
    if orig is None or wm_img is None:
        raise ValueError("Invalid image path.")
    mse = np.mean((orig.astype(np.float64) - wm_img.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def verify_watermark(wm: InvisibleWatermark, image_path: str, expected_text: str) -> dict:
    extracted = wm.extract_watermark(image_path)
    clean_text = "".join(ch for ch in extracted if ch.isprintable())
    return {
        "watermark_present": len(clean_text) > 0,
        "is_authentic": clean_text == expected_text,
        "extracted_text": clean_text,
        "status": "verified" if clean_text == expected_text else "mismatch_or_missing",
    }


if __name__ == "__main__":
    wm = InvisibleWatermark()
    psnr = calculate_psnr("sample.png", "watermarked_output.png")
    result = verify_watermark(wm, "watermarked_output.png", "ImgDNA-Test-2026")
    print("PSNR (dB):", round(psnr, 2))
    print("Verification:", result)