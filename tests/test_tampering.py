"""Run from the project root:  pytest -q"""
import numpy as np
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app
from app.services.ela_forensics import analyze
from app.services.pipeline import VerificationPipeline, TamperingBarrier


@pytest.fixture(scope="module")
def imgs(tmp_path_factory):
    d = tmp_path_factory.mktemp("imgs")
    rng = np.random.default_rng(0)
    base = (np.kron(rng.integers(0, 255, (8, 8, 3)), np.ones((96, 96, 1)))
            + rng.normal(0, 8, (768, 768, 3))).clip(0, 255).astype("uint8")
    Image.fromarray(base).save(d / "ref.jpg", quality=95)
    ref = Image.open(d / "ref.jpg")
    t = ref.copy()
    ImageDraw.Draw(t).rectangle([300, 300, 450, 420], fill=(250, 20, 20))
    t.save(d / "tampered.jpg", quality=80)
    ref.resize((512, 512)).save(d / "rescaled.jpg", quality=70)
    ref.save(d / "ref.png")
    other = rng.integers(0, 255, (6, 6, 3)); other = (np.kron(other, np.ones((128, 128, 1))) + rng.normal(0, 8, (768, 768, 3))).clip(0, 255).astype("uint8")
    Image.fromarray(other).save(d / "other.jpg", quality=95)
    (d / "corrupt.jpg").write_bytes(b"not an image")
    return {k: str(d / v) for k, v in dict(ref="ref.jpg", tampered="tampered.jpg", rescaled="rescaled.jpg",
                                           png="ref.png", corrupt="corrupt.jpg", other="other.jpg").items()}


def test_same_image_is_clean(imgs):
    r = analyze(imgs["ref"], imgs["ref"])
    assert r["barrier"] == "tampering" and r["status"] == "clean"


def test_tampered_detected_with_reference(imgs):
    r = analyze(imgs["tampered"], imgs["ref"])
    assert r["status"] == "tampered" and r["score"] >= 0.9
    box = r["evidence"]["comparison"]["regions"][0]
    # patch sits around x 300-450, y 300-420 of 768 -> normalized ~0.39-0.59 / 0.39-0.55
    assert 0.3 < box["nx"] < 0.5 and 0.3 < box["ny"] < 0.5


def test_rescaled_recompressed_is_not_tampered(imgs):
    assert analyze(imgs["rescaled"], imgs["ref"])["status"] == "clean"


def test_png_ela_is_inconclusive(imgs):
    r = analyze(imgs["png"])
    assert r["status"] == "inconclusive" and r["evidence"]["ela_informative"] is False


def test_png_matching_its_reference_is_clean(imgs):
    # ELA cannot judge a PNG, but the comparison shows it is identical to the reference
    assert analyze(imgs["png"], imgs["ref"])["status"] == "clean"


def test_png_without_reference_stays_inconclusive(imgs):
    assert analyze(imgs["png"])["status"] == "inconclusive"


def test_corrupt_raises(imgs):
    with pytest.raises(ValueError):
        analyze(imgs["corrupt"])


def test_regions_have_normalized_coords(imgs):
    for b in analyze(imgs["tampered"], imgs["ref"])["evidence"]["comparison"]["regions"]:
        assert all(0 <= b[k] <= 1 for k in ("nx", "ny", "nw", "nh"))


# ---- pipeline integration (Person 1's structure + our barrier) ----
def test_pipeline_uses_real_tampering_barrier(imgs):
    p = VerificationPipeline()
    assert isinstance(p.tampering_barrier, TamperingBarrier)
    out = p.run_pipeline(imgs["tampered"], imgs["ref"])
    assert set(out) == {"watermark_result", "deepfake_result", "tampering_result"}
    assert out["tampering_result"]["status"] == "tampered"
    assert out["tampering_result"]["barrier"] == "tampering" and "ela_score" in out["tampering_result"]


def test_pipeline_without_reference(imgs):
    assert VerificationPipeline().run_pipeline(imgs["ref"])["tampering_result"]["status"] == "clean"


def test_corrupt_image_does_not_crash_pipeline(imgs):
    out = VerificationPipeline().run_pipeline(imgs["corrupt"])
    assert out["tampering_result"]["status"] == "error"
    assert out["watermark_result"]["barrier"] == "watermark"      # other barriers still ran
    assert out["deepfake_result"]["barrier"] == "deepfake"


def test_api_endpoint(imgs):
    c = TestClient(app)
    r = c.post("/api/v1/verify", params={"image_path": imgs["tampered"], "reference_path": imgs["ref"]})
    assert r.status_code == 200
    assert r.json()["results"]["tampering_result"]["status"] == "tampered"


# ---- new: wrong reference + quoted paths ----
def test_different_image_is_reference_mismatch_not_tampered(imgs):
    r = analyze(imgs["other"], imgs["ref"])
    assert r["status"] == "reference_mismatch"
    cmp = r["evidence"]["comparison"]
    assert cmp["reference_mismatch"] is True and cmp["regions"] == []


def test_real_tampering_is_not_flagged_as_mismatch(imgs):
    assert analyze(imgs["tampered"], imgs["ref"])["evidence"]["comparison"]["reference_mismatch"] is False


def test_quoted_and_padded_paths_are_cleaned(imgs):
    q = lambda p: '  "' + p + '"  '
    r = analyze(q(imgs["tampered"]), q(imgs["ref"]))
    assert r["status"] == "tampered"


def test_api_with_quoted_paths(imgs):
    c = TestClient(app)
    r = c.post("/api/v1/verify", params={"image_path": '"' + imgs["ref"] + '"'})
    assert r.json()["results"]["tampering_result"]["status"] == "clean"
