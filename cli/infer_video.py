#!/usr/bin/env python3
"""
Video Inference CLI
====================
Video file processing with frame-skip optimization, detection persistence,
and memory-bounded processing.

Usage
-----
    python -m pi_optimized.cli.infer_video --video input.mp4
    python -m pi_optimized.cli.infer_video --video input.mp4 --skip 5 --profile
    python -m pi_optimized.cli.infer_video --camera 0  # Live camera (best-effort)
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
from pi_optimized.core.postprocess import DetectionResult
from pi_optimized.core.profiler import Profiler


def process_video(
    detector: Detector,
    input_path: str,
    output_path: str,
    frame_skip: int = 3,
    profiler: Profiler | None = None,
    max_frames: int | None = None,
) -> None:
    """
    Process video file with frame-skip optimization.

    On skipped frames, the previous detection is reused for annotation,
    providing smooth visual output without redundant inference.

    Parameters
    ----------
    detector : Detector
        Initialized detector.
    input_path : str
        Path to input video file.
    output_path : str
        Path to output annotated video.
    frame_skip : int
        Process every N-th frame (1 = every frame).
    profiler : Profiler, optional
        Profiler instance for benchmarking.
    max_frames : int, optional
        Stop after this many frames (for testing).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {input_path}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0
        logger.warning("Could not read FPS, defaulting to 30")

    logger.info(
        f"Video: {width}×{height} @ {fps:.1f}fps, "
        f"{total_frames} frames, skip={frame_skip}"
    )

    # Video writer
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        logger.error(f"Could not open video writer: {output_path}")
        cap.release()
        return

    frame_idx = 0
    last_result: DetectionResult | None = None
    t_start = time.time()
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            # Decide whether to run inference
            should_detect = (frame_idx % frame_skip == 0) or (last_result is None)

            if should_detect:
                result = detector.detect(frame, frame_index=frame_idx)
                last_result = result

                if profiler:
                    profiler.record(result)
            else:
                # Reuse previous detections for this frame
                result = last_result

            # Compute rolling FPS
            now = time.time()
            current_fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now

            # Annotate
            annotated = detector.annotate(frame, result, fps=current_fps)
            writer.write(annotated)

            frame_idx += 1

            # Progress logging
            if frame_idx % 50 == 0 or frame_idx == total_frames:
                progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                elapsed = time.time() - t_start
                eta = (elapsed / frame_idx * (total_frames - frame_idx)) if frame_idx > 0 else 0
                logger.info(
                    f"  Progress: {frame_idx}/{total_frames} ({progress:.1f}%) "
                    f"ETA: {eta:.0f}s"
                )

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t_start
    actual_fps = frame_idx / elapsed if elapsed > 0 else 0
    logger.info(
        f"Complete: {frame_idx} frames in {elapsed:.1f}s "
        f"({actual_fps:.1f} effective FPS)"
    )
    logger.info(f"Output saved to: {output_path}")


def live_camera(
    detector: Detector,
    camera_index: int = 0,
    profiler: Profiler | None = None,
    frame_skip: int = 3,
) -> None:
    """
    Live camera inference (best-effort, low FPS on Pi3B).

    Parameters
    ----------
    detector : Detector
        Initialized detector.
    camera_index : int
        Camera device index.
    profiler : Profiler, optional
        Profiler instance.
    frame_skip : int
        Process every N-th frame.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Could not open camera {camera_index}")
        return

    # Reduce camera resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    logger.info(
        f"Live camera started (index={camera_index}). "
        f"Press 'q' to quit."
    )

    frame_idx = 0
    last_result: DetectionResult | None = None
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Lost camera connection")
                time.sleep(0.5)
                continue

            should_detect = (frame_idx % frame_skip == 0) or (last_result is None)

            if should_detect:
                result = detector.detect(frame, frame_index=frame_idx)
                last_result = result
                if profiler:
                    profiler.record(result)
            else:
                result = last_result

            now = time.time()
            current_fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now

            annotated = detector.annotate(frame, result, fps=current_fps)
            cv2.imshow("Road Anomaly Detection (q=quit)", annotated)

            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Camera stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if profiler:
        logger.info("\n" + profiler.summary_table())


def main():
    parser = argparse.ArgumentParser(
        description="Road Anomaly Detection — Video/Camera Inference"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", "-v", type=str, help="Input video path")
    group.add_argument("--camera", "-c", type=int, help="Camera index for live feed")

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video path (default: auto-generate)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model path override",
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
        help="Confidence threshold",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=3,
        help="Frame skip interval (default: 3)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling",
    )

    args = parser.parse_args()

    # Build config
    cfg = get_default_config()
    cfg.backend = args.backend
    cfg.input_size = (args.imgsz, args.imgsz)
    cfg.conf_threshold = args.conf
    cfg.frame_skip = args.skip
    cfg.enable_profiling = args.profile

    if args.model:
        if args.backend == "onnx":
            cfg.onnx_export_path = Path(args.model)
        elif args.backend == "ncnn":
            cfg.ncnn_export_dir = Path(args.model)
        else:
            cfg.model_path = Path(args.model)

    detector = Detector(cfg)
    detector.initialize()

    profiler = Profiler() if args.profile else None

    if args.video:
        input_path = Path(args.video)
        if args.output:
            output_path = args.output
        else:
            output_path = str(
                Path("inference_output") / f"{input_path.stem}_detected.mp4"
            )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        process_video(
            detector,
            str(input_path),
            output_path,
            frame_skip=args.skip,
            profiler=profiler,
            max_frames=args.max_frames,
        )

    elif args.camera is not None:
        live_camera(
            detector,
            camera_index=args.camera,
            profiler=profiler,
            frame_skip=args.skip,
        )

    if profiler:
        logger.info("\n=== Profiling Summary ===")
        logger.info(profiler.summary_table())
        csv_path = profiler.export_csv("video_profile.csv")
        logger.info(f"Profile data saved to: {csv_path}")


if __name__ == "__main__":
    main()
