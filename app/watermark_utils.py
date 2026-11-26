import os
import cv2
import numpy as np


class InvisibleWatermark:
    def __init__(self, delimiter: str = "<END>"):
        # علامة نهاية النص لضمان استخراج الرسالة بدقة دون بقايا بتات أخرى
        self.delimiter = delimiter

    def _text_to_bits(self, text: str) -> str:
        full_text = text + self.delimiter
        return "".join(format(byte, "08b") for byte in full_text.encode("utf-8"))

    def _bits_to_text(self, bits: str) -> str:
        delimiter_bits = "".join(
            format(byte, "08b") for byte in self.delimiter.encode("utf-8")
        )
        end_idx = bits.find(delimiter_bits)
        if end_idx != -1:
            bits = bits[:end_idx]

        usable_len = (len(bits) // 8) * 8
        bits = bits[:usable_len]
        byte_array = bytearray(
            int(bits[i : i + 8], 2) for i in range(0, usable_len, 8)
        )
        return byte_array.decode("utf-8", errors="ignore")

    def embed_watermark(
        self, image_path: str, watermark_text: str, output_path: str
    ) -> str:
        """
        Embed a hidden text watermark into the image using LSB technique.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Error: Image path is invalid or image not found.")

        binary_watermark = self._text_to_bits(watermark_text)
        flat_image = image.flatten().astype(np.uint8)

        if len(binary_watermark) > len(flat_image):
            raise ValueError("Error: Watermark is too large for this image.")

        for i in range(len(binary_watermark)):
            bit = int(binary_watermark[i])
            flat_image[i] = np.uint8((int(flat_image[i]) & ~1) | bit)

        watermarked_image = flat_image.reshape(image.shape)

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        success = cv2.imwrite(output_path, watermarked_image)
        if not success:
            raise IOError(f"Error: Failed to save watermarked image to {output_path}")

        return output_path

    def extract_watermark(self, image_path: str) -> str:
        """
        Extract the hidden text watermark from the image using LSB technique.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Error: Image path is invalid or image not found.")

        flat_image = image.flatten().astype(np.uint8)
        delimiter_bits = "".join(
            format(byte, "08b") for byte in self.delimiter.encode("utf-8")
        )

        bits_list = []
        for val in flat_image:
            bits_list.append(str(int(val) & 1))
            if len(bits_list) >= len(delimiter_bits):
                recent = "".join(bits_list[-len(delimiter_bits) :])
                if recent == delimiter_bits:
                    break

        return self._bits_to_text("".join(bits_list))

    def test_robustness(self, watermarked_image_path: str, expected_text: str) -> dict:
        """
        Test watermark extraction under common transformations (JPEG compression and Resize).
        """
        image = cv2.imread(watermarked_image_path)
        if image is None:
            raise ValueError("Error: Image path is invalid or image not found.")

        temp_dir = "temp_robustness_tests"
        os.makedirs(temp_dir, exist_ok=True)
        report = {}

        # 1. اختبار ضغط JPEG بجودات مختلفة
        for quality in [95, 80, 50]:
            jpeg_path = os.path.join(temp_dir, f"q_{quality}.jpg")
            cv2.imwrite(jpeg_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            extracted_jpeg = self.extract_watermark(jpeg_path)
            report[f"jpeg_q{quality}_match"] = extracted_jpeg == expected_text

        # 2. اختبار تغيير الحجم (Resize) ثم إرجاعه للحجم الأصلي
        h, w = image.shape[:2]
        resized_down = cv2.resize(
            image, (max(1, int(w * 0.75)), max(1, int(h * 0.75))), interpolation=cv2.INTER_LINEAR
        )
        resized_back = cv2.resize(resized_down, (w, h), interpolation=cv2.INTER_LINEAR)
        resize_path = os.path.join(temp_dir, "resized_back.png")
        cv2.imwrite(resize_path, resized_back)
        extracted_resize = self.extract_watermark(resize_path)
        report["resize_75pct_back_match"] = extracted_resize == expected_text

        return report