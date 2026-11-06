"""Image DNA - Barrier 3 (pixel-level): preprocessing, ELA, and image comparison.

Key rule: ELA must run on the image at its ORIGINAL resolution. Resizing
destroys the JPEG compression grid that ELA depends on. Only the comparison
step resizes (to align a submitted image with its registered reference).
"""
import io
import os
import re
import uuid
import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity

VERSION = "ela-compare-0.3"


# ---------------------------------------------------------------- preprocessing
def clean_path(p):
    """Windows 'Copy as path' wraps paths in quotes; strip quotes/whitespace. Non-strings pass through."""
    if isinstance(p, str):
        return p.strip().strip('"').strip("'").strip()
    return p


def load_image(path_or_bytes, max_side=4096):
    """Safe load: EXIF rotation, RGB only (drops alpha / converts gray)."""
    path_or_bytes = clean_path(path_or_bytes)
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    try:
        im = Image.open(src)
        im.load()
    except Exception as e:
        raise ValueError(f"Unsupported or corrupt image: {e}")
    fmt = im.format
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > max_side:  # guard against huge uploads
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im, fmt


def to_canonical(im, size=1024):
    """Resize so the longest side == size. Use for comparison/hashing, NOT for ELA."""
    s = size / max(im.size)
    return im.resize((round(im.width * s), round(im.height * s)),
                     Image.LANCZOS if s < 1 else Image.BICUBIC)


# ------------------------------------------------------------------------- ELA
def ela_map(im, quality=90, amplify=None):
    """Return (ela_gray float32 [0..255], ela_rgb uint8 for display)."""
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    re = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(im, np.int16) - np.asarray(re, np.int16)).astype(np.float32)
    gray = diff.max(axis=2)
    if amplify is None:                       # auto-scale so the 99.5th percentile -> 255
        p = np.percentile(gray, 99.5)
        amplify = 255.0 / max(p, 1.0)
    vis = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    return gray, vis


def ela_analyze(im, quality=90, block=16, z_thresh=3.5, texture_norm=True):
    """Block-level robust outlier detection on the ELA map.

    Regions pasted from another source / edited and re-saved usually have a
    different error level than the rest of the image. We flag blocks whose
    mean ELA is an outlier (median/MAD z-score) and return a mask + boxes.
    """
    gray, vis = ela_map(im, quality)
    H, W = gray.shape
    # If the image was last saved at ~the same quality, re-saving changes almost nothing and the
    # map is pure noise: report "uninformative" instead of letting the outlier test amplify it.
    informative = bool(np.percentile(gray, 99) >= 4.0)
    h, w = H // block, W // block
    if h < 4 or w < 4:
        raise ValueError("Image too small for ELA analysis")
    g = gray
    if texture_norm:  # ELA is naturally higher on textured/edge areas; compare like with like
        a = np.asarray(im.convert("L"), np.float32)
        tex = cv2.GaussianBlur(np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)), (0, 0), 8)
        g = gray / (tex + 10.0)
    b = g[:h * block, :w * block].reshape(h, block, w, block).mean(axis=(1, 3))
    med = np.median(b)
    mad = np.median(np.abs(b - med)) * 1.4826 + 1e-3
    z = (b - med) / mad
    blk_mask = (z > z_thresh).astype(np.uint8)
    blk_mask = cv2.morphologyEx(blk_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    blk_mask = cv2.morphologyEx(blk_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.resize(blk_mask, (w * block, h * block), interpolation=cv2.INTER_NEAREST)
    mask = np.pad(mask, ((0, H - mask.shape[0]), (0, W - mask.shape[1])))

    # pixel-level z map (blocks upscaled): feeds per-region scores and the heat-map
    zpix = cv2.resize(z.astype(np.float32), (w * block, h * block), interpolation=cv2.INTER_NEAREST)
    zpix = np.pad(zpix, ((0, H - zpix.shape[0]), (0, W - zpix.shape[1])), mode="edge")

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < 4 * block * block:
            continue
        zr = zpix[y:y + bh, x:x + bw][lab[y:y + bh, x:x + bw] == i]
        boxes.append(dict(x=x, y=y, w=bw, h=bh, area=area,
                          score=round(float(np.clip(zr.max() / 12.0, 0, 1)), 3),
                          mean_z=round(float(zr.mean()), 2)))
    frac = float(mask.mean())
    score = float(np.clip(np.percentile(z, 99.5) / 12.0, 0, 1))
    if not informative:
        mask = np.zeros_like(mask); boxes = []; frac = 0.0; score = 0.0; zpix = np.zeros_like(zpix)
    return dict(informative=informative, ela_vis=vis, ela_gray=gray, mask=mask, boxes=boxes, zmap=zpix,
                flagged_fraction=frac, score=score, max_z=float(z.max()))


def overlay(im, mask, boxes):
    out = np.asarray(im).copy()
    red = np.zeros_like(out); red[..., 0] = 255
    out = np.where(mask[..., None] > 0, (0.6 * out + 0.4 * red).astype(np.uint8), out)
    for b in boxes:
        cv2.rectangle(out, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (255, 255, 0), 2)
    return out


# ------------------------------------------------------------ image comparison
def dhash(im, size=16):
    g = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def hamming_similarity(a, b):
    return 1.0 - float(np.mean(dhash(a) != dhash(b)))


def align_to_reference(sub_bgr, ref_bgr):
    """ORB + RANSAC homography. Handles small shifts/rotations/rescans.
    Falls back to a plain resize when too few matches are found."""
    h, w = ref_bgr.shape[:2]
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), None)
    if d1 is not None and d2 is not None and len(k1) > 20 and len(k2) > 20:
        m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2),
                   key=lambda x: x.distance)[:300]
        if len(m) >= 25:
            src = np.float32([k1[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
            dst = np.float32([k2[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            if H is not None and inl is not None and inl.sum() >= 20:
                return cv2.warpPerspective(sub_bgr, H, (w, h)), True
    return cv2.resize(sub_bgr, (w, h), interpolation=cv2.INTER_AREA), False


def compare_to_reference(sub_im, ref_im, work_side=1024, ssim_thresh=0.6, min_area=1500):
    """Locate differences between a submitted image and its registered original."""
    ref = to_canonical(ref_im, work_side)
    ref_bgr = cv2.cvtColor(np.asarray(ref), cv2.COLOR_RGB2BGR)
    sub_bgr = cv2.cvtColor(np.asarray(sub_im), cv2.COLOR_RGB2BGR)
    gr = cv2.GaussianBlur(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    def _global_ssim(img_bgr):
        g = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        return structural_similarity(gr, g, data_range=255)

    # 1) plain resize first: correct whenever the submission is the same framing (rescaled/recompressed).
    h, w = ref_bgr.shape[:2]
    plain = cv2.resize(sub_bgr, (w, h), interpolation=cv2.INTER_AREA)
    aligned, used_h = plain, False
    # 2) homography only if the plain version is poorly aligned AND the warp really improves it
    #    (a warp on an already-aligned pair just adds interpolation artifacts -> false alarms).
    if _global_ssim(plain) < 0.85:
        warped, ok = align_to_reference(sub_bgr, ref_bgr)
        if ok and _global_ssim(warped) > _global_ssim(plain) + 0.02:
            aligned, used_h = warped, True

    ga = cv2.GaussianBlur(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    ssim, smap = structural_similarity(gr, ga, full=True, data_range=255)
    absdiff = cv2.absdiff(ga, gr)

    mask = ((smap < ssim_thresh) & (absdiff > 25)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        sel = lab[y:y + bh, x:x + bw] == i
        dissim = 1.0 - smap[y:y + bh, x:x + bw][sel]        # 0 = identical, 1+ = very different
        boxes.append(dict(x=x, y=y, w=bw, h=bh, area=area,
                          score=round(float(np.clip(dissim.mean(), 0, 1)), 3)))
    # soft difference map (0..1) in the reference frame: feeds the heat-map when a reference exists
    diff_map = cv2.GaussianBlur(np.clip(1.0 - smap, 0, 1).astype(np.float32) * (absdiff > 25), (0, 0), 3)
    return dict(ssim=float(ssim), hash_similarity=hamming_similarity(sub_im, ref_im), diff_map=diff_map,
                aligned_with_homography=used_h, mask=mask, boxes=boxes,
                changed_fraction=float(mask.mean()),
                aligned_rgb=cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))


# ------------------------------------------------------- API-contract wrapper
MISMATCH_HASH = 0.65       # dHash similarity: ~0.5 for unrelated images, >0.85 for the same image after resize/recompress/local edit
MISMATCH_SSIM = 0.3        # below this AND changed_fraction above MISMATCH_CHANGED -> not the same image
MISMATCH_CHANGED = 0.5     # (initial guesses; calibrate on real data in Week 2)
JPEG_FORMATS = {"JPEG", "MPO"}   # Pillow reports many phone JPEGs as "MPO"


def _with_normalized(boxes, size):
    """Add `space` + normalized (0..1) coordinates so every region is comparable.

    ELA boxes are in the submitted image's pixels; comparison boxes are in the
    resized reference's pixels. The normalized fields (nx, ny, nw, nh) are the
    same for both, so consumers (fusion, heat-map, report) should use them.
    """
    W, H = size
    return [dict(b, nx=round(b["x"] / W, 4), ny=round(b["y"] / H, 4),
                 nw=round(b["w"] / W, 4), nh=round(b["h"] / H, 4)) for b in boxes]


# ----------------------------------------------------- tampered-region analysis
HEAT_Z_MAX = 8.0   # robust z-score that maps to "full red" in the heat-map


def _severity(score):
    return "high" if score >= 0.66 else "medium" if score >= 0.33 else "low"


def _iou(a, b):
    ax2, ay2, bx2, by2 = a["nx"] + a["nw"], a["ny"] + a["nh"], b["nx"] + b["nw"], b["ny"] + b["nh"]
    iw = max(0.0, min(ax2, bx2) - max(a["nx"], b["nx"]))
    ih = max(0.0, min(ay2, by2) - max(a["ny"], b["ny"]))
    inter = iw * ih
    union = a["nw"] * a["nh"] + b["nw"] * b["nh"] - inter
    return inter / union if union > 0 else 0.0


def build_regions(ela_boxes, ela_size, cmp_boxes, cmp_size, iou_merge=0.3):
    """Merge ELA and comparison boxes into one ranked list in normalized (0..1) coordinates.

    Overlapping boxes (IoU > iou_merge) are merged into their union and keep the higher score;
    `sources` says which analyses support the region (more sources = stronger evidence).
    """
    items = []
    for src, boxes, (W, H) in (("ela", ela_boxes, ela_size), ("comparison", cmp_boxes, cmp_size)):
        for b in boxes:
            items.append(dict(nx=b["x"] / W, ny=b["y"] / H, nw=b["w"] / W, nh=b["h"] / H,
                              score=float(b.get("score", 0.0)), sources={src}))
    merged = []
    for it in sorted(items, key=lambda r: -r["score"]):
        for m in merged:
            if _iou(m, it) > iou_merge:
                x1, y1 = min(m["nx"], it["nx"]), min(m["ny"], it["ny"])
                x2 = max(m["nx"] + m["nw"], it["nx"] + it["nw"])
                y2 = max(m["ny"] + m["nh"], it["ny"] + it["nh"])
                m.update(nx=x1, ny=y1, nw=x2 - x1, nh=y2 - y1)
                m["score"] = max(m["score"], it["score"])
                m["sources"] |= it["sources"]
                break
        else:
            merged.append(it)
    merged.sort(key=lambda r: -r["score"])
    return [dict(id=i + 1, sources=sorted(m["sources"]), score=round(m["score"], 3),
                 severity=_severity(m["score"]), area_fraction=round(m["nw"] * m["nh"], 4),
                 nx=round(m["nx"], 4), ny=round(m["ny"], 4), nw=round(m["nw"], 4), nh=round(m["nh"], 4))
            for i, m in enumerate(merged)]


def region_summary(regions, grid=256):
    """Union area of all regions (no double counting) + max score, for the fusion engine."""
    cov = np.zeros((grid, grid), bool)
    for r in regions:
        x1, y1 = int(r["nx"] * grid), int(r["ny"] * grid)
        x2, y2 = int(np.ceil((r["nx"] + r["nw"]) * grid)), int(np.ceil((r["ny"] + r["nh"]) * grid))
        cov[y1:y2, x1:x2] = True
    return dict(region_count=len(regions), flagged_area_fraction=round(float(cov.mean()), 4),
                max_region_score=round(max((r["score"] for r in regions), default=0.0), 3))


# --------------------------------------------------------------- visual outputs
def render_outputs(im, ela, cmp, regions, output_dir, stem):
    """Write overlay / ELA map / heat-map PNGs. Returns file names (relative to output_dir)."""
    os.makedirs(output_dir, exist_ok=True)
    W, H = im.size
    arr = np.asarray(im).copy()

    # 1) overlay: red tint on flagged pixels + numbered boxes (rank = region id)
    mask = (ela["mask"] > 0)
    if cmp is not None and not cmp.get("reference_mismatch"):
        mask = mask | (cv2.resize(cmp["mask"], (W, H), interpolation=cv2.INTER_NEAREST) > 0)
    red = np.zeros_like(arr); red[..., 0] = 255
    ov = np.where(mask[..., None], (0.6 * arr + 0.4 * red).astype(np.uint8), arr)
    thick = max(2, round(max(W, H) / 400))
    for r in regions:
        x1, y1 = int(r["nx"] * W), int(r["ny"] * H)
        x2, y2 = int((r["nx"] + r["nw"]) * W), int((r["ny"] + r["nh"]) * H)
        cv2.rectangle(ov, (x1, y1), (x2, y2), (255, 255, 0), thick)
        cv2.putText(ov, f"#{r['id']} {r['score']:.2f}", (x1 + 3, max(y1 + 18 * thick, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * thick, (255, 255, 0), thick)

    # 2) heat-map: ELA outlier score combined with reference difference (blue = normal, red = strongly anomalous)
    zn = np.clip(ela["zmap"] / HEAT_Z_MAX, 0, 1)
    if cmp is not None and not cmp.get("reference_mismatch"):      # add evidence from the reference comparison
        zn = np.maximum(zn, cv2.resize(cmp["diff_map"], (W, H), interpolation=cv2.INTER_LINEAR))
    heat = cv2.cvtColor(cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    heat = (0.55 * heat + 0.45 * arr).astype(np.uint8)

    files = {}
    for key, img in (("overlay", ov), ("ela_map", ela["ela_vis"]), ("heatmap", heat)):
        name = f"{stem}_{key}.png"
        Image.fromarray(img).save(os.path.join(output_dir, name))
        files[key] = name
    return files


def _make_stem(src):
    base = os.path.splitext(os.path.basename(src))[0] if isinstance(src, str) else "image"
    return re.sub(r"[^A-Za-z0-9_-]", "_", base)[:40] + "_" + uuid.uuid4().hex[:8]


def analyze(sub_path, ref_path=None, output_dir=None):
    """Return the tampering (pixel-level) result in the shared JSON contract.

    status: clean | suspicious | tampered | inconclusive | reference_mismatch
    ELA is only trusted on JPEG input; for PNG etc. the ELA part is
    'inconclusive' and only the comparison with a reference can flag tampering.

    New in 0.3: `evidence.tampered_regions` (merged + ranked, normalized coords, per-region
    score/severity), top-level `region_count`, `flagged_area_fraction`, `max_region_score`, and,
    when `output_dir` is given, PNG file names in `evidence.outputs` (relative to output_dir).
    """
    sub_path = clean_path(sub_path)
    im, fmt = load_image(sub_path)
    is_jpeg = fmt in JPEG_FORMATS
    ela = ela_analyze(im)
    informative = bool(ela["informative"]) and is_jpeg
    if not informative:                       # ELA not trustworthy: no ELA regions / map
        ela = dict(ela, boxes=[], mask=np.zeros_like(ela["mask"]), zmap=np.zeros_like(ela["zmap"]))
    boxes = _with_normalized(ela["boxes"], im.size)
    score = float(ela["score"]) if informative else 0.0
    ev = dict(ela_score=round(score, 3),
              flagged_fraction=round(ela["flagged_fraction"], 4) if informative else 0.0,
              regions=boxes, regions_space="submitted_image_pixels",
              image_size=dict(w=im.width, h=im.height),
              input_format=fmt, ela_informative=informative,
              note=None if is_jpeg else "ELA is not reliable on non-JPEG inputs; ELA result ignored")
    status = ("suspicious" if (boxes and score >= 0.5)
              else "clean" if informative else "inconclusive")
    cmp, cmp_boxes, cmp_size = None, [], (1, 1)
    if ref_path:
        ref_im = load_image(ref_path)[0]
        cmp = compare_to_reference(im, ref_im)
        ref_c = to_canonical(ref_im, 1024)
        # Two unrelated images differ everywhere: that is a wrong reference, not tampering.
        mismatch = (cmp["hash_similarity"] < MISMATCH_HASH
                    or (cmp["ssim"] < MISMATCH_SSIM and cmp["changed_fraction"] > MISMATCH_CHANGED))
        cmp["reference_mismatch"] = bool(mismatch)
        cboxes = [] if mismatch else _with_normalized(cmp["boxes"], ref_c.size)
        ev["comparison"] = dict(ssim=round(cmp["ssim"], 4), hash_similarity=round(cmp["hash_similarity"], 3),
                                changed_fraction=round(cmp["changed_fraction"], 4), regions=cboxes,
                                regions_space="reference_resized_1024_pixels",
                                aligned_with_homography=cmp["aligned_with_homography"],
                                reference_mismatch=bool(mismatch))
        if mismatch:
            status = "reference_mismatch"
            ev["note"] = "Submitted image does not match the reference (different image); tampering not assessed."
        elif cboxes:
            status, score = "tampered", max(score, 0.9)
            cmp_boxes, cmp_size = cboxes, ref_c.size
        elif status == "inconclusive":
            # ELA could not judge (e.g. PNG) but the image matches its reference with no differences
            status = "clean"

    regions = build_regions(ela["boxes"], im.size, cmp_boxes, cmp_size)
    ev["tampered_regions"] = regions
    ev["outputs"] = None
    if output_dir:
        # file names only (relative to output_dir): do not leak server paths through the API
        ev["outputs"] = render_outputs(im, ela, cmp, regions, output_dir, _make_stem(sub_path))
    return dict(barrier="tampering", status=status, score=round(float(score), 3),
                ela_score=ev["ela_score"], **region_summary(regions), evidence=ev, version=VERSION)
