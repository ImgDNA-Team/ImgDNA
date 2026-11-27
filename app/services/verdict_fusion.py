from enum import Enum
from typing import Any, Dict


class Verdict(str, Enum):
    AUTHENTIC = "AUTHENTIC"
    AI_GENERATED = "AI_GENERATED"
    TAMPERED = "TAMPERED"
    INCONCLUSIVE = "INCONCLUSIVE"


class VerdictFusionEngine:
    def fuse(
        self,
        watermark_result: Dict[str, Any],
        deepfake_result: Dict[str, Any],
        tampering_result: Dict[str, Any],
        watermark_is_real: bool = False,
        deepfake_is_real: bool = False,
    ) -> Dict[str, Any]:

        watermark_result = watermark_result or {}
        deepfake_result = deepfake_result or {}
        tampering_result = tampering_result or {}

        reasons = []

        tampering_status = str(tampering_result.get("status", "")).lower()
        deepfake_label = str(deepfake_result.get("label", "")).lower()
        deepfake_error = deepfake_result.get("error")

        if tampering_status == "tampered":
            verdict = Verdict.TAMPERED
            reasons.append(
                "Tampering detector identified strong evidence of manipulation."
            )

        elif deepfake_is_real and deepfake_label == "fake":
            verdict = Verdict.AI_GENERATED
            reasons.append(
                "AI detector classified the image as fake."
            )

        else:
            verdict = Verdict.INCONCLUSIVE

            if tampering_status == "reference_mismatch":
                reasons.append(
                    "The submitted image does not match the reference image."
                )

            elif tampering_status == "suspicious":
                reasons.append(
                    "Tampering detector found suspicious regions."
                )

            elif tampering_status == "error":
                reasons.append(
                    "Tampering detector failed."
                )

            if deepfake_error:
                reasons.append(
                    "AI detector failed: " + str(deepfake_error)
                )

            elif not deepfake_is_real:
                reasons.append(
                    "AI detector is not yet verified as a real detector."
                )

            elif deepfake_label == "suspicious":
                reasons.append(
                    "AI detector found suspicious AI-related patterns."
                )

            elif deepfake_label == "real":
                reasons.append(
                    "AI detector found no manipulation evidence."
                )

            if not watermark_is_real:
                reasons.append(
                    "Watermark detector is not yet verified as a real detector."
                )

            if not reasons:
                reasons.append(
                    "Available evidence is insufficient to determine authenticity."
                )

        return {
            "verdict": verdict.value,
            "confidence_score": None,
            "reasons": reasons,
            "barriers_results": {
                "watermark": watermark_result,
                "deepfake": deepfake_result,
                "tampering": tampering_result,
            },
            "message": "Verdict generated based on available detector results."
        }