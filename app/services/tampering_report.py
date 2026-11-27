"""Presentation layer for the tampering barrier (Person 3).

Turns the raw barrier result into what the API returns to the user: a compact summary,
the ranked suspicious regions, links to the visual outputs, and one human-readable sentence
for the verdict's `reasons`. It only reads fields, so it tolerates error results and old shapes.
"""
from typing import Any, Dict, Optional

OUTPUTS_URL = "/outputs"          # where main.py serves the PNGs


def summarize_tampering(tampering_result: Optional[Dict[str, Any]], outputs_url: str = OUTPUTS_URL) -> Dict[str, Any]:
    tr = tampering_result or {}
    ev = tr.get("evidence") or {}
    regions = ev.get("tampered_regions") or []
    outputs = ev.get("outputs") or None
    return dict(
        status=tr.get("status"),
        score=tr.get("score"),
        region_count=tr.get("region_count", len(regions)),
        flagged_area_fraction=tr.get("flagged_area_fraction", 0.0),
        max_region_score=tr.get("max_region_score", 0.0),
        suspicious_regions=regions,
        output_images={k: f"{outputs_url.rstrip('/')}/{v}" for k, v in outputs.items()} if outputs else None,
        note=ev.get("note"),
    )


def _where(r: Dict[str, Any]) -> str:
    """'upper-left', 'center', ... from the region's centre (normalized coordinates)."""
    cx, cy = r["nx"] + r["nw"] / 2, r["ny"] + r["nh"] / 2
    v = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
    h = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    return "center" if (v, h) == ("middle", "center") else f"{v}-{h}" if v != "middle" else h


def describe_regions(tampering_result: Optional[Dict[str, Any]]) -> Optional[str]:
    """One sentence for the verdict reasons, or None when there is nothing to point at."""
    tr = tampering_result or {}
    if tr.get("status") not in ("tampered", "suspicious"):
        return None
    regions = (tr.get("evidence") or {}).get("tampered_regions") or []
    if not regions:
        return None
    top = regions[0]
    n = len(regions)
    return (f"{n} suspicious region{'s' if n != 1 else ''} covering {tr.get('flagged_area_fraction', 0) * 100:.1f}% "
            f"of the image; strongest is in the {_where(top)} (score {top['score']:.2f}, {top['severity']}).")
