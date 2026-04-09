#!/usr/bin/env python3
"""
Image Inference CLI
====================
Single image or batch directory inference for road anomaly detection.
Outputs annotated images + JSON detection files.

Usage
-----
    python -m pi_optimized.cli.infer_image --image test.jpg
    python -m pi_optimized.cli.infer_image --dir ./test_images/ --output ./results/
    python -m pi_optimized.cli.infer_image --image test.jpg --backend pytorch --profile
"""

import argparse
import sys
import time
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))

from pi_optimized.config import PipelineConfig, get_default_config
from pi_optimized.core.detector import Detector
from pi_optimized.core.profiler import Profiler

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def process_single_image(
    detector: Detector,
    image_path: Path,
    output_dir: Path,
    profiler: Profiler | None = None,
    save_json: bool = True,
) -> None:
    """Process a single image and save results."""
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Could not read image: {image_path}")
        return

    logger.info(f"Processing: {image_path.name} ({img.shape[1]}×{img.shape[0]})")

    # Detect
    result = detector.detect(img)

    # Profile
    if profiler:
        profiler.record(result)

    # Annotate
    annotated = detector.annotate(img, result)

    # Save annotated image
    out_img = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
    cv2.imwrite(str(out_img), annotated)
    logger.info(f"  Saved: {out_img.name}")

    # Save JSON
    if save_json:
        out_json = output_dir / f"{image_path.stem}_detections.json"
        with open(out_json, "w") as f:
            f.write(result.to_json())

    # Log results
    logger.info(
        f"  Detections: {result.count} | "
        f"Inference: {result.inference_ms:.1f}ms | "
        f"Total: {result.total_ms:.1f}ms"
    )
    for det in result.detections:
        logger.info(
            f"    → {det.class_name} ({det.severity}) "
            f"confidence={det.confidence:.3f} "
            f"bbox={det.bbox}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Road Anomaly Detection — Image Inference"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i", type=str, help="Single image path")
    group.add_argument("--dir", "-d", type=str, help="Directory of images")

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="inference_output",
        help="Output directory (default: inference_output)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model path (ONNX or .pt). Defaults to config.",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["onnx", "ncnn", "pytorch"],
        default="onnx",
        help="Inference backend (default: onnx)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Input resolution (default: 320)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (default: 0.35)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling (memory + latency)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON output",
    )

    args = parser.parse_args()

    # Build config
    cfg = get_default_config()
    cfg.backend = args.backend
    cfg.input_size = (args.imgsz, args.imgsz)
    cfg.conf_threshold = args.conf
    cfg.enable_profiling = args.profile

    if args.model:
        if args.backend == "onnx":
            cfg.onnx_export_path = Path(args.model)
        elif args.backend == "ncnn":
            cfg.ncnn_export_dir = Path(args.model)
        else:
            cfg.model_path = Path(args.model)

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector(cfg)
    detector.initialize()

    profiler = Profiler(export_dir=output_dir) if args.profile else None

    # Gather images
    if args.image:
        images = [Path(args.image)]
    else:
        img_dir = Path(args.dir)
        images = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        logger.info(f"Found {len(images)} images in {img_dir}")

    if not images:
        logger.error("No images found!")
        sys.exit(1)

    # Process
    t_start = time.time()
    for img_path in images:
        process_single_image(
            detector, img_path, output_dir,
            profiler=profiler,
            save_json=not args.no_json,
        )

    elapsed = time.time() - t_start
    logger.info(f"\nProcessed {len(images)} images in {elapsed:.1f}s")

    # Profiling summary
    if profiler:
        logger.info("\n=== Profiling Summary ===")
        logger.info(profiler.summary_table())
        csv_path = profiler.export_csv("image_profile.csv")
        logger.info(f"Profile data saved to: {csv_path}")


if __name__ == "__main__":
    main()
