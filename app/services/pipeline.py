from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from app.services.ela_forensics import analyze as tampering_analyze
from app.services.verdict_fusion import VerdictFusionEngine
from app.services.tampering_report import summarize_tampering, describe_regions

from pathlib import Path
# from AI_Detector.ai_detector import DeepfakeDetector

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

    _detector = None

    def _get_detector(self):
        if DeepfakeBarrier._detector is None:
            from AI_Detector.ai_detector import DeepfakeDetector

            root = Path(__file__).resolve().parents[2]
            model_dir = root / "AI_Detector"

            DeepfakeBarrier._detector = DeepfakeDetector(
                model_dir / "best.pth",
                model_dir / "thresholds.json"
            )

        return DeepfakeBarrier._detector

    def process(self, image_path: str,
                reference_path: Optional[str] = None) -> Dict[str, Any]:

        detector = self._get_detector()

        result = detector.predict(
            image_path,
            with_heatmap=True
        )

        return result

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
            deepfake_is_real=True,
        )

        # Point the user at WHERE the tampering evidence is (Person 3)
        reasons = list(fusion_result["reasons"])
        region_reason = describe_regions(results["tampering_result"])
        if region_reason:
            reasons.append(region_reason)

        return {
            "status": "success",
            "results": results,
            "verdict": fusion_result["verdict"],
            "confidence_score": fusion_result["confidence_score"],
            "reasons": reasons,
            "message": fusion_result["message"],
            "tampering_analysis": summarize_tampering(results["tampering_result"]),
        }
    
