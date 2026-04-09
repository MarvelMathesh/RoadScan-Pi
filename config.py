"""
Pi-Optimized Road Anomaly Detection — Configuration
=====================================================
Central configuration module. All tuneable parameters live here.
Designed for Raspberry Pi 3B (1GB RAM, ARM Cortex-A53, no GPU).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PROJECT_ROOT.parent  # top-level repo

# ---------------------------------------------------------------------------
# Class taxonomy — matches the 7-class custom-trained YOLOv8m
# ---------------------------------------------------------------------------
CLASS_NAMES: dict[int, str] = {
    0: "Heavy-Vehicle",
    1: "Light-Vehicle",
    2: "Pedestrian",
    3: "Crack",
    4: "Crack-Severe",
    5: "Pothole",
    6: "Speed-Bump",
}

# Severity tiers for UI colour coding
SEVERITY_MAP: dict[str, str] = {
    "Heavy-Vehicle": "info",
    "Light-Vehicle": "info",
    "Pedestrian": "info",
    "Crack": "warning",
    "Crack-Severe": "critical",
    "Pothole": "critical",
    "Speed-Bump": "warning",
}

# Detection box colours (BGR for OpenCV)
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 140, 0),    # Heavy-Vehicle  — orange
    1: (0, 200, 0),      # Light-Vehicle  — green
    2: (200, 200, 0),    # Pedestrian     — cyan
    3: (0, 180, 255),    # Crack          — amber
    4: (0, 0, 255),      # Crack-Severe   — red
    5: (0, 50, 255),     # Pothole        — deep red
    6: (255, 200, 0),    # Speed-Bump     — sky blue
}


@dataclass
class PipelineConfig:
    """
    Central configuration for the entire detection pipeline.

    Attributes
    ----------
    model_path : Path
        Path to model file (.onnx, .param+.bin, or .pt).
    backend : str
        Inference backend — 'onnx', 'ncnn', or 'pytorch'.
    input_size : tuple[int, int]
        Model input resolution (width, height).
    conf_threshold : float
        Minimum detection confidence.
    iou_threshold : float
        IoU threshold for NMS.
    max_detections : int
        Maximum detections per frame after NMS.
    frame_skip : int
        For video: process every N-th frame (1 = every frame).
    roi_crop : bool
        If True, crop to bottom 65% of frame (road surface focus).
    enable_profiling : bool
        If True, log per-component latency and memory stats.
    num_threads : int
        Number of inference threads (ONNX Runtime / NCNN).
    memory_limit_mb : int
        Soft memory budget — triggers warnings if exceeded.
    """

    # ---- Model ----
    model_path: Path = field(
        default_factory=lambda: _REPO_ROOT
        / "RoadDetectionModel"
        / "RoadModel_yolov8m.pt_rounds120_b9"
        / "weights"
        / "best.pt"
    )
    backend: str = "onnx"  # 'onnx' | 'ncnn' | 'pytorch'
    input_size: tuple[int, int] = (320, 320)
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_detections: int = 50

    # ---- Video ----
    frame_skip: int = 3
    roi_crop: bool = False
    roi_top_fraction: float = 0.35  # crop top 35%

    # ---- Performance ----
    enable_profiling: bool = False
    num_threads: int = 4
    memory_limit_mb: int = 700

    # ---- Export ----
    onnx_export_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "best_road.onnx"
    )
    ncnn_export_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "best_road_ncnn"
    )

    # ---- Web server ----
    upload_folder: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "app" / "static" / "uploads"
    )
    result_folder: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "app" / "static" / "results"
    )
    max_upload_mb: int = 50

    # ---- Class info (read-only) ----
    class_names: dict[int, str] = field(default_factory=lambda: CLASS_NAMES.copy())
    class_colors: dict[int, tuple[int, int, int]] = field(
        default_factory=lambda: CLASS_COLORS.copy()
    )
    severity_map: dict[str, str] = field(
        default_factory=lambda: SEVERITY_MAP.copy()
    )

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def resolve_model_path(self) -> Path:
        """Return the correct model path for the selected backend."""
        if self.backend == "onnx":
            return self.onnx_export_path
        elif self.backend == "ncnn":
            return self.ncnn_export_dir
        return self.model_path

    def ensure_directories(self) -> None:
        """Create required output directories."""
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        self.result_folder.mkdir(parents=True, exist_ok=True)
        self.onnx_export_path.parent.mkdir(parents=True, exist_ok=True)
        self.ncnn_export_dir.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        valid_backends = {"onnx", "ncnn", "pytorch"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"Invalid backend '{self.backend}'. Choose from {valid_backends}"
            )


# ---------------------------------------------------------------------------
# Singleton default config
# ---------------------------------------------------------------------------
def get_default_config() -> PipelineConfig:
    """Return a default config, overridable via environment variables."""
    cfg = PipelineConfig()

    # Allow env-var overrides for common settings
    if env_backend := os.environ.get("RAD_BACKEND"):
        cfg.backend = env_backend
    if env_res := os.environ.get("RAD_INPUT_SIZE"):
        size = int(env_res)
        cfg.input_size = (size, size)
    if env_conf := os.environ.get("RAD_CONF_THRESHOLD"):
        cfg.conf_threshold = float(env_conf)
    if env_threads := os.environ.get("RAD_NUM_THREADS"):
        cfg.num_threads = int(env_threads)
    if env_skip := os.environ.get("RAD_FRAME_SKIP"):
        cfg.frame_skip = int(env_skip)

    return cfg
