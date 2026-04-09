#!/usr/bin/env python3
"""
Lightweight Flask Server — Pi-Optimized
==========================================
Designed for Raspberry Pi 3B: single model instance, minimal memory,
no Streamlit/WebRTC overhead.

Endpoints:
    GET  /                — Upload page
    POST /detect/image    — Image detection (returns results page)
    POST /detect/video    — Video detection with SSE progress
    GET  /api/health      — Health check with memory stats
    POST /api/detect      — Headless API (returns JSON)

Usage
-----
    python -m pi_optimized.app.server
    RAD_BACKEND=pytorch python -m pi_optimized.app.server --port 8080
"""

import os
import sys
import time
import uuid
import json
import tempfile
import logging
from pathlib import Path
from functools import wraps

import cv2
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    jsonify,
    Response,
    stream_with_context,
)
from werkzeug.utils import secure_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))

from pi_optimized.config import PipelineConfig, get_default_config
from pi_optimized.core.detector import Detector
from pi_optimized.core.profiler import Profiler

# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.secret_key = os.urandom(24)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}

# Global detector (loaded once at startup)
_detector: Detector | None = None
_config: PipelineConfig | None = None
_profiler = Profiler()


def get_detector() -> Detector:
    global _detector, _config
    if _detector is None:
        _config = get_default_config()
        _config.ensure_directories()
        _detector = Detector(_config)
        _detector.initialize()
        logger.info(f"Detector initialized: {_detector}")
    return _detector


def get_config() -> PipelineConfig:
    global _config
    if _config is None:
        get_detector()
    return _config


def allowed_file(filename: str, allowed_ext: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_ext


def get_memory_mb() -> float:
    """Get current process memory usage."""
    return _profiler.get_memory_mb()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Main upload page."""
    mem = get_memory_mb()
    return render_template(
        "index.html",
        memory_mb=f"{mem:.0f}",
        backend=get_config().backend,
        input_size=get_config().input_size[0],
        conf_threshold=get_config().conf_threshold,
    )


@app.route("/detect/image", methods=["POST"])
def detect_image():
    """Process uploaded image and return results page."""
    if "file" not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXT):
        flash("Invalid file type. Supported: JPG, PNG, BMP, WebP", "error")
        return redirect(url_for("index"))

    try:
        cfg = get_config()
        detector = get_detector()

        # Parse optional confidence override
        conf = request.form.get("confidence", None)
        if conf:
            conf = float(conf)

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        upload_path = cfg.upload_folder / unique_name
        file.save(str(upload_path))

        # Read and detect
        img = cv2.imread(str(upload_path))
        if img is None:
            flash("Could not decode image file", "error")
            return redirect(url_for("index"))

        result = detector.detect(img, conf_threshold=conf)
        _profiler.record(result)

        # Annotate and save
        annotated = detector.annotate(img, result)
        result_name = f"result_{unique_name}"
        result_path = cfg.result_folder / result_name
        cv2.imwrite(str(result_path), annotated)

        # Prepare detection info for template
        detections_info = []
        for i, det in enumerate(result.detections):
            detections_info.append({
                "id": i + 1,
                "class_name": det.class_name,
                "confidence": f"{det.confidence:.2%}",
                "severity": det.severity,
                "bbox": det.bbox,
            })

        return render_template(
            "results.html",
            original=unique_name,
            result=result_name,
            detections=detections_info,
            stats={
                "total_detections": result.count,
                "critical": len(result.by_severity("critical")),
                "warning": len(result.by_severity("warning")),
                "info": len(result.by_severity("info")),
                "inference_ms": f"{result.inference_ms:.0f}",
                "total_ms": f"{result.total_ms:.0f}",
                "memory_mb": f"{get_memory_mb():.0f}",
                "image_size": f"{img.shape[1]}×{img.shape[0]}",
            },
        )

    except Exception as e:
        logger.error(f"Image detection error: {e}", exc_info=True)
        flash(f"Processing error: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/detect/video", methods=["POST"])
def detect_video():
    """Process uploaded video file. Returns SSE progress stream."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename, ALLOWED_VIDEO_EXT):
        return jsonify({"error": "Invalid video format"}), 400

    cfg = get_config()
    detector = get_detector()

    # Save uploaded video
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    upload_path = cfg.upload_folder / unique_name
    file.save(str(upload_path))

    # Parse options
    frame_skip = int(request.form.get("frame_skip", cfg.frame_skip))
    conf = request.form.get("confidence", None)
    if conf:
        conf = float(conf)

    def generate():
        """Generator for SSE progress stream."""
        cap = cv2.VideoCapture(str(upload_path))
        if not cap.isOpened():
            yield f"data: {json.dumps({'error': 'Could not open video'})}\n\n"
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video
        result_name = f"result_{Path(unique_name).stem}.mp4"
        result_path = cfg.result_folder / result_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(result_path), fourcc, fps, (width, height)
        )

        frame_idx = 0
        last_result = None
        t_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                should_detect = (
                    frame_idx % frame_skip == 0 or last_result is None
                )

                if should_detect:
                    result = detector.detect(
                        frame, conf_threshold=conf, frame_index=frame_idx
                    )
                    last_result = result
                else:
                    result = last_result

                annotated = detector.annotate(frame, result)
                writer.write(annotated)

                frame_idx += 1

                # Send progress every 10 frames
                if frame_idx % 10 == 0 or frame_idx == total_frames:
                    progress = (
                        frame_idx / total_frames * 100
                        if total_frames > 0
                        else 0
                    )
                    elapsed = time.time() - t_start
                    eta = (
                        (elapsed / frame_idx * (total_frames - frame_idx))
                        if frame_idx > 0
                        else 0
                    )
                    yield f"data: {json.dumps({'progress': round(progress, 1), 'frame': frame_idx, 'total': total_frames, 'eta': round(eta), 'detections': result.count})}\n\n"

        finally:
            cap.release()
            writer.release()

        # Send completion event
        elapsed = time.time() - t_start
        yield f"data: {json.dumps({'complete': True, 'result_file': result_name, 'elapsed': round(elapsed, 1), 'frames': frame_idx})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/download/<filename>")
def download_result(filename):
    """Download a processed result file."""
    cfg = get_config()
    return send_from_directory(str(cfg.result_folder), filename, as_attachment=True)


@app.route("/api/health")
def health():
    """Health check with system stats."""
    return jsonify({
        "status": "ok",
        "memory_mb": round(get_memory_mb(), 1),
        "backend": get_config().backend,
        "input_size": get_config().input_size,
        "cpu_temp": _profiler.get_cpu_temp(),
    })


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Headless detection API.
    POST an image file and get JSON detections back.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    detector = get_detector()
    conf = request.form.get("confidence", None)
    if conf:
        conf = float(conf)

    result = detector.detect(img, conf_threshold=conf)
    return jsonify(result.to_dict())


@app.route("/static/uploads/<path:filename>")
def serve_upload(filename):
    cfg = get_config()
    return send_from_directory(str(cfg.upload_folder), filename)


@app.route("/static/results/<path:filename>")
def serve_result(filename):
    cfg = get_config()
    return send_from_directory(str(cfg.result_folder), filename)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def create_app(backend: str = None, model_path: str = None) -> Flask:
    """Factory for creating the Flask app with custom config."""
    if backend:
        os.environ["RAD_BACKEND"] = backend
    # Pre-load detector on startup
    with app.app_context():
        get_detector()
    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Road Anomaly Detection — Web Server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--backend", choices=["onnx", "ncnn", "pytorch"], default=None
    )
    parser.add_argument("--model", type=str, default=None, help="Model path override")

    args = parser.parse_args()

    if args.backend:
        os.environ["RAD_BACKEND"] = args.backend
    if args.model:
        os.environ["RAD_MODEL_PATH"] = args.model

    logger.info("=" * 50)
    logger.info("Road Anomaly Detection — Pi-Optimized Server")
    logger.info("=" * 50)

    # Pre-load detector
    get_detector()
    logger.info(f"Memory after model load: {get_memory_mb():.0f} MB")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
