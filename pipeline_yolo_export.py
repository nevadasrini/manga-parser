"""Map YOLO training image variants to :mod:`src.panel_pipeline` configs (IntroCV cv/)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Final

import cv2
import numpy as np

# cv package lives at repo_root/cv; panel_pipeline imports ``src.*``.
_CV_ROOT = Path(__file__).resolve().parents[1] / "cv"
if str(_CV_ROOT) not in sys.path:
    sys.path.insert(0, str(_CV_ROOT))

from src.panel_pipeline import PanelPipelineConfig, run_panel_pipeline  # noqa: E402

RAW: Final = "raw"
PREPROCESS_ONLY: Final = "preprocess_only"
PREPROCESS_EDGES: Final = "preprocess_edges"
PREPROCESS_REPAIR_EDGES: Final = "preprocess_repair_edges"
EDGES_REPAIR_ONLY: Final = "edges_repair_only"

IMAGE_VARIANT_CHOICES: Final[tuple[str, ...]] = (
    RAW,
    PREPROCESS_ONLY,
    PREPROCESS_EDGES,
    PREPROCESS_REPAIR_EDGES,
    EDGES_REPAIR_ONLY,
)


def _edge_export_config(
    *,
    use_preprocessing: bool,
    use_border_repair: bool,
) -> PanelPipelineConfig:
    """Canny edge map as YOLO input; skip contour work (not needed for pixels)."""
    return PanelPipelineConfig(
        use_preprocessing=use_preprocessing,
        use_panel_detection=True,
        use_blur_before_canny=True,
        use_canny=True,
        use_border_repair=use_border_repair,
        use_contours=False,
        use_geometry_filter=False,
        use_polygon_approximation=False,
    )


def panel_config_for_variant(variant: str) -> PanelPipelineConfig:
    if variant == PREPROCESS_ONLY:
        return PanelPipelineConfig(
            use_preprocessing=True,
            use_panel_detection=False,
        )
    if variant == PREPROCESS_EDGES:
        return _edge_export_config(use_preprocessing=True, use_border_repair=False)
    if variant == PREPROCESS_REPAIR_EDGES:
        return _edge_export_config(use_preprocessing=True, use_border_repair=True)
    if variant == EDGES_REPAIR_ONLY:
        return _edge_export_config(use_preprocessing=False, use_border_repair=True)
    raise ValueError(f"unknown image variant: {variant!r}")


VARIANT_LABELS: Dict[str, str] = {
    RAW: "original JPEG (no classical pipeline)",
    PREPROCESS_ONLY: "preprocess only (grayscale signal for detection)",
    PREPROCESS_EDGES: "preprocess + Canny edges",
    PREPROCESS_REPAIR_EDGES: "preprocess + edge repair (dilate/close) + Canny",
    EDGES_REPAIR_ONLY: "edge repair + Canny on raw gray (no preprocess)",
}


def bgr_for_yolo_variant(image_bgr: np.ndarray, variant: str) -> np.ndarray:
    """
    Return BGR uint8 image to write as YOLO training/val image.

    ``variant`` must be one of :data:`IMAGE_VARIANT_CHOICES` except ``raw`` (pass-through).
    """
    if variant == RAW:
        return image_bgr
    cfg = panel_config_for_variant(variant)
    res = run_panel_pipeline(image_bgr, config=cfg)
    if variant == PREPROCESS_ONLY:
        g = res.gray_for_detection
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if variant == PREPROCESS_EDGES:
        e = res.edges
        return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
    if variant in (PREPROCESS_REPAIR_EDGES, EDGES_REPAIR_ONLY):
        e = res.edges_closed
        return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unknown image variant: {variant!r}")
