"""Week 2: tampered-region analysis + visual outputs.  Run: pytest -q"""
import numpy as np
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app
from app.services.ela_forensics import analyze, build_regions, region_summary
from app.services.pipeline import VerificationPipeline


@pytest.fixture(scope="module")
def imgs(tmp_path_factory):
    d = tmp_path_factory.mktemp("w2")
    rng = np.random.default_rng(1)
    base = (np.kron(rng.integers(0, 255, (8, 8, 3)), np.ones((96, 96, 1)))
            + rng.normal(0, 8, (768, 768, 3))).clip(0, 255).astype("uint8")
    Image.fromarray(base).save(d / "ref.jpg", quality=95)
    ref = Image.open(d / "ref.jpg")
    t = ref.copy()
    ImageDraw.Draw(t).rectangle([300, 300, 450, 420], fill=(250, 20, 20))
    t.save(d / "tampered.jpg", quality=80)
    other = (np.kron(rng.integers(0, 255, (6, 6, 3)), np.ones((128, 128, 1)))
             + rng.normal(0, 8, (768, 768, 3))).clip(0, 255).astype("uint8")
    Image.fromarray(other).save(d / "other.jpg", quality=95)
    ref.save(d / "ref.png")
    return {k: str(d / v) for k, v in dict(ref="ref.jpg", tampered="tampered.jpg", other="other.jpg",
                                           png="ref.png").items()} | {"dir": str(d)}


# ------------------------------------------------------------ region analysis
def test_regions_have_score_severity_and_sources(imgs):
    r = analyze(imgs["tampered"], imgs["ref"])
    regs = r["evidence"]["tampered_regions"]
    assert regs and r["region_count"] == len(regs)
    for g in regs:
        assert 0 <= g["score"] <= 1 and g["severity"] in {"low", "medium", "high"}
        assert set(g["sources"]) <= {"ela", "comparison"}
        assert all(0 <= g[k] <= 1 for k in ("nx", "ny", "nw", "nh", "area_fraction"))
    assert [g["id"] for g in regs] == list(range(1, len(regs) + 1))
    assert [g["score"] for g in regs] == sorted((g["score"] for g in regs), reverse=True)


def test_top_region_sits_on_the_edited_patch(imgs):
    top = analyze(imgs["tampered"], imgs["ref"])["evidence"]["tampered_regions"][0]
    cx, cy = top["nx"] + top["nw"] / 2, top["ny"] + top["nh"] / 2
    assert 0.35 < cx < 0.65 and 0.35 < cy < 0.6          # patch centre is ~ (0.49, 0.47)
    assert "comparison" in top["sources"]


def test_summary_fields_for_fusion(imgs):
    r = analyze(imgs["tampered"], imgs["ref"])
    assert 0 < r["flagged_area_fraction"] < 0.5 and r["max_region_score"] > 0.3


def test_clean_image_has_no_regions(imgs):
    r = analyze(imgs["ref"], imgs["ref"])
    assert r["region_count"] == 0 and r["flagged_area_fraction"] == 0 and r["max_region_score"] == 0


def test_reference_mismatch_has_no_comparison_regions(imgs):
    r = analyze(imgs["other"], imgs["ref"])
    assert r["status"] == "reference_mismatch"
    assert all("comparison" not in g["sources"] for g in r["evidence"]["tampered_regions"])


def test_overlapping_boxes_are_merged_and_union_area_not_double_counted():
    ela = [dict(x=100, y=100, w=100, h=100, score=0.4)]
    cmp_ = [dict(x=110, y=110, w=100, h=100, score=0.8)]
    regs = build_regions(ela, (1000, 1000), cmp_, (1000, 1000))
    assert len(regs) == 1 and regs[0]["sources"] == ["comparison", "ela"] and regs[0]["score"] == 0.8
    far = build_regions(ela, (1000, 1000), [dict(x=700, y=700, w=100, h=100, score=0.9)], (1000, 1000))
    assert len(far) == 2 and far[0]["score"] == 0.9                    # ranked by score
    twice = region_summary(build_regions([], (1, 1), [], (1, 1)) or [dict(nx=0, ny=0, nw=.5, nh=.5, score=.5)] * 2)
    assert twice["flagged_area_fraction"] == pytest.approx(0.25, abs=0.01)


# ------------------------------------------------------------- visual outputs
def test_no_output_dir_means_no_files(imgs):
    assert analyze(imgs["tampered"], imgs["ref"])["evidence"]["outputs"] is None


def test_outputs_are_written_and_have_image_size(imgs, tmp_path):
    r = analyze(imgs["tampered"], imgs["ref"], output_dir=str(tmp_path))
    out = r["evidence"]["outputs"]
    for key in ("overlay", "ela_map", "heatmap"):
        im = Image.open(tmp_path / out[key])
        assert im.format == "PNG" and im.size == (768, 768)
    ov = np.asarray(Image.open(tmp_path / out["overlay"]).convert("RGB")).astype(int)
    assert ((ov[..., 0] > 200) & (ov[..., 1] > 200) & (ov[..., 2] < 80)).any()   # yellow boxes drawn


def test_outputs_for_png_and_mismatch_do_not_crash(imgs, tmp_path):
    for sub, ref in ((imgs["png"], None), (imgs["other"], imgs["ref"])):
        out = analyze(sub, ref, output_dir=str(tmp_path))["evidence"]["outputs"]
        assert (tmp_path / out["heatmap"]).exists()


def test_two_runs_do_not_overwrite_each_other(imgs, tmp_path):
    a = analyze(imgs["ref"], output_dir=str(tmp_path))["evidence"]["outputs"]["overlay"]
    b = analyze(imgs["ref"], output_dir=str(tmp_path))["evidence"]["outputs"]["overlay"]
    assert a != b


# ---------------------------------------------------------- pipeline / API
def test_pipeline_passes_output_dir_only_to_tampering(imgs, tmp_path):
    out = VerificationPipeline().run_pipeline(imgs["tampered"], imgs["ref"], output_dir=str(tmp_path))
    assert out["tampering_result"]["evidence"]["outputs"]["overlay"]
    assert out["watermark_result"]["barrier"] == "watermark" and out["deepfake_result"]["barrier"] == "deepfake"


def test_api_save_outputs_and_serve_image(imgs):
    c = TestClient(app)
    r = c.post("/api/v1/verify", params={"image_path": imgs["tampered"], "reference_path": imgs["ref"],
                                         "save_outputs": True})
    name = r.json()["results"]["tampering_result"]["evidence"]["outputs"]["heatmap"]
    g = c.get(f"/outputs/{name}")
    assert g.status_code == 200 and g.headers["content-type"] == "image/png"


def test_api_default_does_not_write_files(imgs):
    c = TestClient(app)
    r = c.post("/api/v1/verify", params={"image_path": imgs["tampered"]})
    assert r.json()["results"]["tampering_result"]["evidence"]["outputs"] is None


def test_heatmap_is_hotter_on_the_edit_than_on_the_background(imgs, tmp_path):
    """The reference comparison must show up in the heat-map even when ELA finds nothing."""
    out = analyze(imgs["tampered"], imgs["ref"], output_dir=str(tmp_path))["evidence"]["outputs"]
    plain = analyze(imgs["tampered"], output_dir=str(tmp_path))["evidence"]["outputs"]   # no reference
    h = np.asarray(Image.open(tmp_path / out["heatmap"]).convert("RGB")).astype(int)
    h0 = np.asarray(Image.open(tmp_path / plain["heatmap"]).convert("RGB")).astype(int)
    patch = (slice(310, 410), slice(310, 440))
    assert np.abs(h[patch] - h0[patch]).mean() > 10      # reference adds evidence inside the patch


def test_outputs_do_not_expose_server_paths(imgs, tmp_path):
    out = analyze(imgs["tampered"], imgs["ref"], output_dir=str(tmp_path))["evidence"]["outputs"]
    assert set(out) == {"overlay", "ela_map", "heatmap"}
    assert all("/" not in v and "\\\\" not in v for v in out.values())
