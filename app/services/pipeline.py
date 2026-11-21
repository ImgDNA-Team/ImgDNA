from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from app.services.ela_forensics import analyze as tampering_analyze
from app.services.verdict_fusion import VerdictFusionEngine

class VerificationBarrier(ABC):
    name = "barrier"

    @abstractmethod
    def process(self, image_path: str, reference_path: Optional[str] = None) -> Dict[str, Any]:
        pass


class WatermarkBarrier(VerificationBarrier):
    name = "watermark"

    def process(self, image_path: str, reference_path: Optional[str] = None) -> Dict[str, Any]:
        return {"barrier": "watermark", "status": "intact", "payload_matched": True}


class DeepfakeBarrier(VerificationBarrier):
    name = "deepfake"

    def process(self, image_path: str, reference_path: Optional[str] = None) -> Dict[str, Any]:
        return {"barrier": "deepfake", "manipulation_score": 0.05}


class TamperingBarrier(VerificationBarrier):
    """Pixel-level barrier (Person 3): preprocessing + ELA + comparison with the registered original."""
    name = "tampering"

    def process(self, image_path: str, reference_path: Optional[str] = None,
                output_dir: Optional[str] = None) -> Dict[str, Any]:
        # output_dir (optional): where to write the overlay / ELA map / heat-map PNGs
        return tampering_analyze(image_path, reference_path, output_dir=output_dir)


class VerificationPipeline:
    def __init__(self):
        self.watermark_barrier = WatermarkBarrier()
        self.deepfake_barrier = DeepfakeBarrier()
        self.tampering_barrier = TamperingBarrier()

    @staticmethod
    def _safe_run(barrier: VerificationBarrier, image_path: str, reference_path: Optional[str],
                  **kwargs) -> Dict[str, Any]:
        """One failing barrier must not crash the others (NFR-06)."""
        try:
            return barrier.process(image_path, reference_path, **kwargs)
        except Exception as e:
            return {"barrier": barrier.name, "status": "error", "message": str(e)}

    def run_pipeline(self, image_path: str, reference_path: Optional[str] = None,
                     output_dir: Optional[str] = None) -> Dict[str, Any]:
        return {
            "watermark_result": self._safe_run(self.watermark_barrier, image_path, reference_path),
            "deepfake_result": self._safe_run(self.deepfake_barrier, image_path, reference_path),
            "tampering_result": self._safe_run(self.tampering_barrier, image_path, reference_path,
                                                  output_dir=output_dir),
        }

    def run_with_verdict(
        self,
        image_path: str,
        reference_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:

        # Run all verification barriers
        results = self.run_pipeline(
            image_path,
            reference_path,
            output_dir
        )

        # Fuse the results into one final verdict
        fusion_engine = VerdictFusionEngine()

        fusion_result = fusion_engine.fuse(
            watermark_result=results["watermark_result"],
            deepfake_result=results["deepfake_result"],
            tampering_result=results["tampering_result"],
            watermark_is_real=False,
            deepfake_is_real=False,
        )

        return {
            "status": "success",
            "results": results,
            "verdict": fusion_result["verdict"],
            "confidence_score": fusion_result["confidence_score"],
            "reasons": fusion_result["reasons"],
            "message": fusion_result["message"],
        }
    
