from abc import ABC, abstractmethod
from typing import Dict, Any

class VerificationBarrier(ABC):
    @abstractmethod
    def process(self, image_path: str) -> Dict[str, Any]:
        pass

class WatermarkBarrier(VerificationBarrier):
    def process(self, image_path: str) -> Dict[str, Any]:
        return {"barrier": "watermark", "status": "intact", "payload_matched": True}

class DeepfakeBarrier(VerificationBarrier):
    def process(self, image_path: str) -> Dict[str, Any]:
        return {"barrier": "deepfake", "manipulation_score": 0.05}

class TamperingBarrier(VerificationBarrier):
    def process(self, image_path: str) -> Dict[str, Any]:
        return {"barrier": "tampering", "ela_score": 0.02}

class VerificationPipeline:
    def __init__(self):
        self.watermark_barrier = WatermarkBarrier()
        self.deepfake_barrier = DeepfakeBarrier()
        self.tampering_barrier = TamperingBarrier()

    def run_pipeline(self, image_path: str) -> Dict[str, Any]:
        res_b1 = self.watermark_barrier.process(image_path)
        res_b2a = self.deepfake_barrier.process(image_path)
        res_b2b = self.tampering_barrier.process(image_path)

        return {
            "watermark_result": res_b1,
            "deepfake_result": res_b2a,
            "tampering_result": res_b2b
        }