#!/usr/bin/env python3
"""
Benchmark Suite
================
Automated benchmarking for research-grade evaluation of the detection
pipeline across different configurations.

Usage
-----
    python -m pi_optimized.cli.benchmark --model models/best_road.onnx
    python -m pi_optimized.cli.benchmark --model best.pt --backend pytorch --resolutions 320 416 640
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))

from pi_optimized.config import PipelineConfig, get_default_config
from pi_optimized.core.detector import Detector
from pi_optimized.core.profiler import Profiler


def benchmark_resolution(
    model_path: str,
    backend: str,
    resolution: int,
    num_iterations: int = 100,
    warmup: int = 10,
    num_threads: int = 4,
) -> dict:
    """
    Benchmark a single resolution configuration.

    Parameters
    ----------
    model_path : str
        Path to model file.
    backend : str
        Inference backend.
    resolution : int
        Input resolution (square).
    num_iterations : int
        Number of inference iterations.
    warmup : int
        Warmup iterations (excluded from timing).
    num_threads : int
        Number of threads.

    Returns
    -------
    dict
        Benchmark results.
    """
    cfg = get_default_config()
    cfg.backend = backend
    cfg.input_size = (resolution, resolution)
    cfg.num_threads = num_threads
    cfg.enable_profiling = True

    if backend == "onnx":
        cfg.onnx_export_path = Path(model_path)
    elif backend == "ncnn":
        cfg.ncnn_export_dir = Path(model_path)
    else:
        cfg.model_path = Path(model_path)

    detector = Detector(cfg)
    detector.initialize()

    profiler = Profiler()

    # Generate a realistic test image (random noise doesn't represent real data well,
    # but gives consistent timing)
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup
    logger.info(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        detector.detect(test_img)

    # Benchmark
    logger.info(f"  Running {num_iterations} iterations...")
    profiler.reset()

    for i in range(num_iterations):
        result = detector.detect(test_img, frame_index=i)
        profiler.record(result)

    summary = profiler.summary()
    summary["resolution"] = resolution
    summary["backend"] = backend
    summary["num_threads"] = num_threads
    summary["model_path"] = model_path

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Road Anomaly Detection — Benchmark Suite"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model path (ONNX, NCNN dir, or .pt)",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["onnx", "ncnn", "pytorch"],
        default="onnx",
        help="Inference backend",
    )
    parser.add_argument(
        "--resolutions", "-r",
        type=int,
        nargs="+",
        default=[320],
        help="Resolutions to benchmark (default: 320)",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of inference iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=4,
        help="Number of threads (default: 4)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results",
        help="Output directory",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    logger.info("=" * 60)
    logger.info("ROAD ANOMALY DETECTION — BENCHMARK SUITE")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Resolutions: {args.resolutions}")
    logger.info(f"Iterations: {args.iterations} (+{args.warmup} warmup)")
    logger.info(f"Threads: {args.threads}")
    logger.info("=" * 60)

    for res in args.resolutions:
        logger.info(f"\n--- Resolution: {res}×{res} ---")
        try:
            result = benchmark_resolution(
                model_path=args.model,
                backend=args.backend,
                resolution=res,
                num_iterations=args.iterations,
                warmup=args.warmup,
                num_threads=args.threads,
            )
            all_results.append(result)

            # Print summary for this resolution
            inf = result["inference_ms"]
            logger.info(
                f"  Inference: mean={inf['mean']:.1f}ms "
                f"p50={inf['p50']:.1f}ms "
                f"p95={inf['p95']:.1f}ms "
                f"p99={inf['p99']:.1f}ms"
            )
            logger.info(
                f"  FPS: mean={result['fps']['mean']:.2f} "
                f"max={result['fps']['max']:.2f}"
            )
            logger.info(
                f"  Memory: peak={result['memory_mb']['peak']:.1f}MB"
            )

        except Exception as e:
            logger.error(f"  Benchmark failed for {res}: {e}")
            import traceback
            traceback.print_exc()

    # ---- Summary table ----
    if all_results:
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK RESULTS SUMMARY")
        logger.info("=" * 60)

        # Markdown table
        lines = [
            "| Resolution | Backend | Inference (mean) | Inference (p95) | FPS (mean) | Memory (peak) |",
            "|------------|---------|------------------|-----------------|------------|---------------|",
        ]
        for r in all_results:
            lines.append(
                f"| {r['resolution']}×{r['resolution']} "
                f"| {r['backend']} "
                f"| {r['inference_ms']['mean']:.1f} ms "
                f"| {r['inference_ms']['p95']:.1f} ms "
                f"| {r['fps']['mean']:.2f} "
                f"| {r['memory_mb']['peak']:.1f} MB |"
            )

        table = "\n".join(lines)
        logger.info("\n" + table)

        # Save results
        json_path = output_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {json_path}")

        md_path = output_dir / "benchmark_results.md"
        with open(md_path, "w") as f:
            f.write("# Benchmark Results\n\n")
            f.write(f"**Model:** `{args.model}`\n\n")
            f.write(f"**Backend:** {args.backend}\n\n")
            f.write(f"**Iterations:** {args.iterations}\n\n")
            f.write(table + "\n")
        logger.info(f"Markdown table saved to: {md_path}")


if __name__ == "__main__":
    main()
