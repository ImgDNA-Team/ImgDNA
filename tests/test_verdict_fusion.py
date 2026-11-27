import pytest

from app.services.verdict_fusion import VerdictFusionEngine, Verdict


@pytest.fixture
def engine():
    return VerdictFusionEngine()


def test_tampered_image(engine):
    result = engine.fuse(
        watermark_result={"status": "intact"},
        deepfake_result={"label": "real"},
        tampering_result={"status": "tampered"},
        watermark_is_real=True,
        deepfake_is_real=True,
    )

    assert result["verdict"] == Verdict.TAMPERED.value


def test_ai_generated_image(engine):
    result = engine.fuse(
        watermark_result={"status": "intact"},
        deepfake_result={"label": "fake"},
        tampering_result={"status": "clean"},
        watermark_is_real=True,
        deepfake_is_real=True,
    )

    assert result["verdict"] == Verdict.AI_GENERATED.value


def test_suspicious_image_is_inconclusive(engine):
    result = engine.fuse(
        watermark_result={"status": "intact"},
        deepfake_result={"label": "suspicious"},
        tampering_result={"status": "suspicious"},
        watermark_is_real=True,
        deepfake_is_real=True,
    )

    assert result["verdict"] == Verdict.INCONCLUSIVE.value


def test_reference_mismatch_is_not_tampered(engine):
    result = engine.fuse(
        watermark_result={"status": "intact"},
        deepfake_result={"label": "real"},
        tampering_result={"status": "reference_mismatch"},
        watermark_is_real=True,
        deepfake_is_real=True,
    )

    assert result["verdict"] == Verdict.INCONCLUSIVE.value


def test_placeholder_detectors_cannot_confirm_authenticity(engine):
    result = engine.fuse(
        watermark_result={"status": "intact"},
        deepfake_result={"label": "real"},
        tampering_result={"status": "clean"},
        watermark_is_real=False,
        deepfake_is_real=False,
    )

    assert result["verdict"] == Verdict.INCONCLUSIVE.value


def test_error_does_not_mean_authentic(engine):
    result = engine.fuse(
        watermark_result={"status": "error"},
        deepfake_result={"label": "real"},
        tampering_result={"status": "error"},
        watermark_is_real=False,
        deepfake_is_real=False,
    )

    assert result["verdict"] == Verdict.INCONCLUSIVE.value

def test_pipeline_run_with_verdict():
    from app.services.pipeline import VerificationPipeline

    pipeline = VerificationPipeline()

    result = pipeline.run_with_verdict(
        image_path="dummy_image.jpg"
    )

    assert result["status"] == "success"
    assert result["verdict"] == Verdict.INCONCLUSIVE.value
    assert result["confidence_score"] is None
    assert "watermark_result" in result["results"]
    assert "deepfake_result" in result["results"]
    assert "tampering_result" in result["results"]