from app.watermark_utils import InvisibleWatermark

wm = InvisibleWatermark()

input_path = "sample.png"
output_path = "watermarked_output.png"
secret_text = "ImgDNA-Test-2026"

wm.embed_watermark(input_path, secret_text, output_path)
print("Embedded successfully:", output_path)

if hasattr(wm, "extract_watermark"):
  print("Extracted:", wm.extract_watermark(output_path))

if hasattr(wm, "test_robustness"):
  print("Robustness:", wm.test_robustness(output_path, secret_text))