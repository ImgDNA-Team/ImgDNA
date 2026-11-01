from typing import Optional
from fastapi import FastAPI
from app.services.pipeline import VerificationPipeline

app = FastAPI(title="Image DNA Verification API", version="1.1")
pipeline = VerificationPipeline()

@app.get("/")
def read_root():
    return {"message": "Welcome to Image DNA Verification Pipeline - Developed by shahd_elhewehi"}

@app.post("/api/v1/verify")
def verify_image(image_path: str, reference_path: Optional[str] = None):
    results = pipeline.run_pipeline(image_path, reference_path)
    return {
        "status": "success",
        "results": results
    }