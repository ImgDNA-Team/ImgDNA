import cv2
import numpy as np

# ImgDNA Project - Person 2: Invisible Watermarking Module
# Developed for handling image invisible watermarking embedding and extraction.

class InvisibleWatermark:
    def __init__(self):
        # Initializing basic settings for watermark processing
        pass

    def embed_watermark(self, image_path, watermark_text, output_path):
        """
        Embed a hidden text watermark into the image using LSB technique.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Error: Image path is invalid or image not found.")
            
        # Convert text to binary format
        binary_watermark = ''.join(format(ord(i), '08b') for i in watermark_text)
        binary_watermark += '1111101011001110' # Custom delimiter to mark the end of the text
        
        flat_image = image.flatten()
        
        if len(binary_watermark) > len(flat_image):
            raise ValueError("Error: The watermark text is too large for this image!")
            
        # Modify the least significant bit of pixels
        for i in range(len(binary_watermark)):
            flat_image[i] = (flat_image[i] & ~1) | int(binary_watermark[i])
            
        watermarked_image = flat_image.reshape(image.shape)
        cv2.imwrite(output_path, watermarked_image)
        print(f"Success: Watermarked image saved at {output_path}")

    def extract_watermark(self, image_path):
        """
        Extract the hidden text watermark from the image.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Error: Image path is invalid or image not found.")
            
        flat_image = image.flatten()
        
        binary_data = ""
        # Extract the LSB from pixels
        for i in range(len(flat_image)):
            binary_data += str(flat_image[i] & 1)
            
        # Group bits into bytes
        all_bytes = [binary_data[i: i+8] for i in range(0, len(binary_data), 8)]
        
        decoded_text = ""
        for byte in all_bytes:
            decoded_text += chr(int(byte, 2))
            if len(decoded_text) > 100:  # Safety limit for extraction
                break
                
        return decoded_text

if __name__ == "__main__":
    # Test run for the watermark utility
    wm = InvisibleWatermark()
    # Example usage:
    # wm.embed_watermark("input.jpg", "ImgDNA_Secure", "output.jpg")
    # print(wm.extract_watermark("output.jpg"))