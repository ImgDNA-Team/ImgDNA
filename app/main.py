from typing import Optional
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.services.pipeline import VerificationPipeline

app = FastAPI(title="Image DNA Verification API", version="1.1")
pipeline = VerificationPipeline()

# Tampered-region images (overlay / ELA map / heat-map) are written here and served under /outputs
OUTPUT_DIR = os.environ.get("IMGDNA_OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/")
def read_root():
    return {"message": "Welcome to Image DNA Verification Pipeline - Developed by shahd_elhewehi"}

@app.post("/api/v1/verify")
def verify_image(image_path: str, reference_path: Optional[str] = None, save_outputs: bool = False):
    results = pipeline.run_pipeline(image_path, reference_path,
                                    output_dir=OUTPUT_DIR if save_outputs else None)
    return {
        "status": "success",
        "results": results
    }