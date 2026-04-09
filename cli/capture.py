#!/usr/bin/env python3
"""
USB Camera Capture & Inference CLI
==================================
Native V4L2 support for USB cameras on Raspberry Pi.
Allows capturing a single image or recording a short video clip,
and immediately running road anomaly inference on it.

Usage
-----
    # Capture a single image and run inference
    python -m pi_optimized.cli.capture --mode image --camera 0

    # Record a 5-second video clip and run inference
    python -m pi_optimized.cli.capture --mode video --duration 5 --camera 0
"""

import argparse
import sys
import time
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))

from pi_optimized.config import PipelineConfig, get_default_config
from pi_optimized.core.detector import Detector
from pi_optimized.core.profiler import Profiler
from pi_optimized.cli.infer_video import process_video


def get_v4l2_camera(index: int, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """Initialize camera with V4L2 backend for Linux/Pi."""
    logger.info(f"Initializing V4L2 Camera (Index {index}) at {width}x{height}...")
    
    # Use V4L2 specifically, important for Raspberry Pi USB cameras
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera index {index}.")
        sys.exit(1)

    # Set MJPEG or YUYV if necessary, but letting OpenCV choose is usually fine
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Verify settings
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f"Camera opened successfully. Resolution: {actual_w}x{actual_h}")
    
    return cap


def capture_and_infer_image(detector: Detector, camera_index: int, output_dir: Path):
    """Capture a single image from USB camera and run inference."""
    cap = get_v4l2_camera(camera_index)
    
    # Warm-up: discard first few frames to let auto-exposure/white-balance settle
    logger.info("Warming up camera sensor...")
    for _ in range(15):
        cap.grab()
        time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logger.error("Failed to capture image from camera.")
        return

    # Generate timestamped filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"capture_{timestamp}.jpg"
    out_path = output_dir / f"capture_{timestamp}_detected.jpg"

    # Save raw image
    cv2.imwrite(str(raw_path), frame)
    logger.info(f"Raw image saved to {raw_path}")

    # Run inference
    logger.info("Running detection...")
    result = detector.detect(frame)
    annotated = detector.annotate(frame, result)

    # Save result
    cv2.imwrite(str(out_path), annotated)
    logger.info(f"Detected image saved to {out_path}")
    logger.info(f"Found {result.count} anomalies.")


def capture_and_infer_video(detector: Detector, camera_index: int, duration: int, output_dir: Path):
    """Record a video clip from USB camera then run inference."""
    cap = get_v4l2_camera(camera_index)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60:
        fps = 30.0  # Fallback
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"record_{timestamp}.mp4"
    out_path = output_dir / f"record_{timestamp}_detected.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))
    
    if not writer.isOpened():
        logger.error("Failed to initialize video writer.")
        cap.release()
        return

    logger.info("Warming up camera sensor...")
    for _ in range(15):
        cap.grab()
        time.sleep(0.05)
        
    logger.info(f"Recording video for {duration} seconds...")
    start_time = time.time()
    frames_recorded = 0
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Dropped frame during recording.")
            continue
        writer.write(frame)
        frames_recorded += 1
        
    cap.release()
    writer.release()
    
    logger.info(f"Recording complete. Saved {frames_recorded} frames to {raw_path}")
    
    # Now run inference on the saved video using our existing pipeline
    logger.info(f"Running offline detection on recorded video...")
    process_video(
        detector=detector,
        input_path=str(raw_path),
        output_path=str(out_path),
        frame_skip=3  # Standard frame skip for Pi optimization
    )


def main():
    parser = argparse.ArgumentParser(description="USB Camera V4L2 Capture & Inference")
    parser.add_argument("--mode", choices=["image", "video"], required=True, 
                        help="Capture a single 'image' or record a 'video' clip.")
    parser.add_argument("--camera", type=int, default=0, help="V4L2 Camera Index (default: 0)")
    parser.add_argument("--duration", type=int, default=5, 
                        help="Duration in seconds for video mode (default: 5)")
    parser.add_argument("--output", type=str, default="inference_output", 
                        help="Output directory")
    
    # Model Options
    parser.add_argument("--model", type=str, default=None, help="Model path override")
    parser.add_argument("--backend", choices=["onnx", "ncnn", "pytorch"], default="onnx",
                        help="Inference backend (default: onnx)")
    parser.add_argument("--imgsz", type=int, default=320, help="Input resolution")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")

    args = parser.parse_args()

    # Prep output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Init Config & Detector
    cfg = get_default_config()
    cfg.backend = args.backend
    cfg.input_size = (args.imgsz, args.imgsz)
    cfg.conf_threshold = args.conf
    
    if args.model:
        if args.backend == "onnx":
            cfg.onnx_export_path = Path(args.model)
        else:
            cfg.model_path = Path(args.model)

    detector = Detector(cfg)
    detector.initialize()

    if args.mode == "image":
        capture_and_infer_image(detector, args.camera, output_dir)
    elif args.mode == "video":
        capture_and_infer_video(detector, args.camera, args.duration, output_dir)


if __name__ == "__main__":
    main()
