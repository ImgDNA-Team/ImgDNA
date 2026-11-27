import cv2
import numpy as np
from app.watermark_utils import InvisibleWatermark


def embed_in_array(wm: InvisibleWatermark, image: np.ndarray, text: str) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array.")
    bits = wm._text_to_bits(text)
    flat = image.flatten().astype(np.uint8).copy()
    if len(bits) > len(flat):
        raise ValueError("Watermark is too large for this image.")
    for i, bit in enumerate(bits):
        flat[i] = np.uint8((int(flat[i]) & ~1) | int(bit))
    return flat.reshape(image.shape)


def extract_from_array(wm: InvisibleWatermark, image: np.ndarray) -> str:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array.")
    delimiter_bits = "".join(
        format(byte, "08b") for byte in wm.delimiter.encode("utf-8")
    )
    bits_list = []
    for val in image.flatten().astype(np.uint8):
        bits_list.append(str(int(val) & 1))
        if len(bits_list) >= len(delimiter_bits):
            if "".join(bits_list[-len(delimiter_bits) :]) == delimiter_bits:
                break
    return wm._bits_to_text("".join(bits_list))


if __name__ == "__main__":
    wm = InvisibleWatermark()
    img = np.random.randint(10, 240, (100, 100, 3), dtype=np.uint8)

    # 1. اختبار التضمين والاستخراج في الذاكرة (Array)
    wm_img = embed_in_array(wm, img, "ImgDNA-W4-Mem")
    extracted = extract_from_array(wm, wm_img)
    print("1. Array test:", extracted == "ImgDNA-W4-Mem", "| Extracted:", extracted)

    # 2. اختبار صورة بدون بصمة (أو نص غير مطابق)
    clean_extracted = extract_from_array(wm, img)
    print("2. Clean image extracted:", repr(clean_extracted))

    # 3. اختبار نص أطول من سعة الصورة
    try:
        embed_in_array(wm, img, "X" * 50000)
        print("3. Overflow test: FAILED (no error raised)")
    except ValueError as e:
        print("3. Overflow test: PASSED |", e)

    # 4. اختبار صورة غير صالحة
    try:
        embed_in_array(wm, np.array([]), "test")
        print("4. Invalid array test: FAILED (no error raised)")
    except ValueError as e:
        print("4. Invalid array test: PASSED |", e)