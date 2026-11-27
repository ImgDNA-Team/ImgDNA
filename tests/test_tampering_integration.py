"""Week 3: tampering barrier inside the full pipeline (verdict + suspicious regions) and via the API."""
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

import app.services.ela_forensics as ef
from app.main import app
from app.services.ela_forensics import analyze
from app.services.pipeline import VerificationPipeline
from app.services.tampering_report import summarize_tampering, describe_regions, _where


@pytest.fixture(scope="module")
def imgs(tmp_path_factory):
    d = tmp_path_factory.mktemp("w3")
    rng = np.random.default_rng(2)
    base = (np.kron(rng.integers(0, 255, (8, 8, 3)), np.ones((96, 96, 1)))
            + rng.normal(0, 8, (768, 768, 3))).clip(0, 255).astype("uint8")
    Image.fromarray(base).save(d / "ref.jpg", quality=95)
    ref = Image.open(d / "ref.jpg")
    t = ref.copy()
    ImageDraw.Draw(t).rectangle([300, 300, 450, 420], fill=(250, 20, 20))
    t.save(d / "tampered.jpg", quality=80)
    (d / "corrupt.jpg").write_bytes(b"not an image")
    return {k: str(d / v) for k, v in dict(ref="ref.jpg", tampered="tampered.jpg", corrupt="corrupt.jpg").items()}


@pytest.fixture
def ela_says_suspicious(monkeypatch):
    """Force ELA to report one strong outlier box, whatever the image is."""
    real = ef.ela_analyze

    def fake(im, *a, **k):
        r = real(im, *a, **k)
        m = np.zeros_like(r["mask"]); m[100:196, 100:196] = 1
        z = np.zeros_like(r["zmap"]); z[100:196, 100:196] = 9.0
        r.update(informative=True, mask=m, zmap=z, score=0.7, flagged_fraction=float(m.mean()),
                 boxes=[dict(x=100, y=100, w=96, h=96, area=96 * 96, score=0.8, mean_z=9.0)])
        return r
    monkeypatch.setattr(ef, "ela_analyze", fake)


# ----------------------------------------------------------- verdict response
def test_tampered_image_returns_verdict_and_regions(imgs):
    out = VerificationPipeline().run_with_verdict(imgs["tampered"], imgs["ref"])
    assert out["verdict"] == "TAMPERED"
    ta = out["tampering_analysis"]
    assert ta["status"] == "tampered" and ta["region_count"] == len(ta["suspicious_regions"]) >= 1
    top = ta["suspicious_regions"][0]
    assert 0.3 < top["nx"] + top["nw"] / 2 < 0.7 and "comparison" in top["sources"]
    assert any("suspicious region" in r for r in out["reasons"])            # human sentence for the user
    assert ta["output_images"] is None                                      # images are opt-in


def test_output_images_are_urls_to_existing_files(imgs, tmp_path):
    out = VerificationPipeline().run_with_verdict(imgs["tampered"], imgs["ref"], output_dir=str(tmp_path))
    urls = out["tampering_analysis"]["output_images"]
    assert set(urls) == {"overlay", "ela_map", "heatmap"}
    for u in urls.values():
        assert u.startswith("/outputs/") and os.path.exists(tmp_path / u.split("/")[-1])


def test_unmodified_image_has_no_region_sentence(imgs):
    out = VerificationPipeline().run_with_verdict(imgs["ref"], imgs["ref"])
    assert out["tampering_analysis"]["status"] == "clean" and out["tampering_analysis"]["suspicious_regions"] == []
    assert not any("suspicious region" in r for r in out["reasons"])


def test_broken_image_still_gives_a_verdict(imgs):
    out = VerificationPipeline().run_with_verdict(imgs["corrupt"])
    assert out["status"] == "success" and out["verdict"] == "INCONCLUSIVE"
    ta = out["tampering_analysis"]
    assert ta["status"] == "error" and ta["suspicious_regions"] == [] and ta["output_images"] is None


# ------------------------------------------------------------------- API
def test_api_returns_verdict_and_tampering_analysis(imgs):
    c = TestClient(app)
    r = c.post("/api/v1/verify", params={"image_path": imgs["tampered"], "reference_path": imgs["ref"],
                                         "save_outputs": True})
    j = r.json()
    assert r.status_code == 200 and j["verdict"] == "TAMPERED"
    url = j["tampering_analysis"]["output_images"]["overlay"]
    g = c.get(url)
    assert g.status_code == 200 and g.headers["content-type"] == "image/png"
    assert j["results"]["tampering_result"]["status"] == "tampered"          # old shape still there


# ------------------------------------------- reference beats ELA-only suspicion
def test_ela_alone_is_only_suspicious(imgs, ela_says_suspicious):
    r = analyze(imgs["ref"])
    assert r["status"] == "suspicious" and r["region_count"] == 1
    out = VerificationPipeline().run_with_verdict(imgs["ref"])
    assert out["verdict"] == "INCONCLUSIVE"                                  # never an accusation by ELA alone
    assert any("strongest is in the" in x for x in out["reasons"])


def test_matching_reference_overrides_ela_outliers(imgs, ela_says_suspicious, tmp_path):
    r = analyze(imgs["ref"], imgs["ref"], output_dir=str(tmp_path))
    assert r["status"] == "clean" and r["score"] == 0.0 and r["region_count"] == 0
    assert "ignored" in r["evidence"]["note"] and r["evidence"]["tampered_regions"] == []
    # the picture agrees with the verdict: nothing is tinted red
    ov = np.asarray(Image.open(tmp_path / r["evidence"]["outputs"]["overlay"]).convert("RGB")).astype(int)
    orig = np.asarray(Image.open(imgs["ref"]).convert("RGB")).astype(int)
    assert np.abs(ov - orig).mean() < 1.0


def test_real_edit_is_still_tampered_when_ela_is_suspicious(imgs, ela_says_suspicious):
    assert analyze(imgs["tampered"], imgs["ref"])["status"] == "tampered"


# ---------------------------------------------------- report helper (pure)
def test_describe_regions_only_for_suspicious_or_tampered():
    reg = dict(id=1, nx=0.05, ny=0.05, nw=0.2, nh=0.2, score=0.91, severity="high", sources=["ela"])
    base = dict(flagged_area_fraction=0.04, evidence=dict(tampered_regions=[reg]))
    s = describe_regions(dict(base, status="tampered"))
    assert "1 suspicious region covering 4.0%" in s and "upper-left" in s and "0.91" in s and "high" in s
    assert describe_regions(dict(base, status="suspicious")) is not None
    for st in ("clean", "inconclusive", "reference_mismatch", "error", None):
        assert describe_regions(dict(base, status=st)) is None
    assert describe_regions(dict(status="tampered", evidence={})) is None
    assert describe_regions(None) is None


def test_summarize_tolerates_error_and_empty_results():
    for tr in (None, {}, {"status": "error", "message": "boom"}):
        s = summarize_tampering(tr)
        assert s["suspicious_regions"] == [] and s["output_images"] is None and s["region_count"] == 0


def test_where_words():
    box = lambda cx, cy: dict(nx=cx - .05, ny=cy - .05, nw=.1, nh=.1)
    assert _where(box(.5, .5)) == "center" and _where(box(.1, .1)) == "upper-left"
    assert _where(box(.9, .9)) == "lower-right" and _where(box(.9, .5)) == "right" and _where(box(.5, .1)) == "upper-center"


def test_comparison_backed_regions_rank_above_ela_only_ones():
    from app.services.ela_forensics import build_regions
    ela = [dict(x=700, y=700, w=100, h=100, score=0.95)]                     # strong-looking ELA outlier elsewhere
    cmp_ = [dict(x=100, y=100, w=100, h=100, score=0.40)]                    # real difference vs the original
    regs = build_regions(ela, (1000, 1000), cmp_, (1000, 1000))
    assert regs[0]["sources"] == ["comparison"] and regs[1]["sources"] == ["ela"] and regs[0]["id"] == 1
