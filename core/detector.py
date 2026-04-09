"""
Multi-Backend Detector
=======================
Unified inference interface supporting ONNX Runtime, NCNN, and PyTorch backends.
ONNX Runtime is the primary backend for Raspberry Pi deployment.
"""

import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from pi_optimized.config import PipelineConfig, get_default_config
from pi_optimized.core.preprocess import preprocess, rescale_detections
from pi_optimized.core.postprocess import (
    Detection,
    DetectionResult,
    parse_yolov8_output,
    draw_detections,
    draw_stats_overlay,
)

logger = logging.getLogger(__name__)


class BaseBackend:
    """Abstract inference backend."""

    def __init__(self, model_path: str, num_threads: int = 4):
        self.model_path = model_path
        self.num_threads = num_threads

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def warmup(self, input_shape: tuple) -> None:
        """Run a dummy inference to warm up the runtime."""
        dummy = np.random.randn(*input_shape).astype(np.float32)
        self.run(dummy)


class ONNXBackend(BaseBackend):
    """ONNX Runtime inference backend — primary for Pi deployment."""

    def __init__(self, model_path: str, num_threads: int = 4):
        super().__init__(model_path, num_threads)
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Install with: "
                "pip install onnxruntime"
            )

        # Configure session options for ARM efficiency
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1  # Single-socket, reduce overhead
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        # Prefer CPU execution provider
        providers = ["CPUExecutionProvider"]

        logger.info(f"Loading ONNX model: {model_path}")
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info(
            f"ONNX model loaded. Input: {self.input_name}, "
            f"Outputs: {self.output_names}"
        )

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run(
            self.output_names, {self.input_name: input_tensor}
        )
        return outputs[0]


class NCNNBackend(BaseBackend):
    """NCNN inference backend — best ARM NEON performance."""

    def __init__(self, model_path: str, num_threads: int = 4):
        super().__init__(model_path, num_threads)
        try:
            import ncnn
        except ImportError:
            raise ImportError(
                "ncnn python bindings not installed. "
                "See https://github.com/Tencent/ncnn"
            )

        model_dir = Path(model_path)
        param_path = str(model_dir / "model.ncnn.param")
        bin_path = str(model_dir / "model.ncnn.bin")

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = num_threads
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        logger.info(f"NCNN model loaded from {model_dir}")

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        import ncnn

        # NCNN expects (C, H, W) without batch dimension
        mat_in = ncnn.Mat(input_tensor[0])
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        _, mat_out = ex.extract("out0")
        return np.array(mat_out).reshape(1, -1, mat_out.w)


class PyTorchBackend(BaseBackend):
    """
    PyTorch/Ultralytics backend — for desktop development only.

    WARNING: Consumes ~500MB+ RAM. Not suitable for Pi3B deployment.
    """

    def __init__(self, model_path: str, num_threads: int = 4):
        super().__init__(model_path, num_threads)
        logger.warning(
            "PyTorch backend selected. This uses ~500MB+ RAM and is "
            "NOT recommended for Raspberry Pi deployment."
        )
        try:
            import torch

            torch.set_num_threads(num_threads)
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "PyTorch/Ultralytics not installed. "
                "pip install torch ultralytics"
            )

        self._yolo = YOLO(model_path)
        self._class_names = self._yolo.names
        logger.info(f"PyTorch model loaded: {model_path}")

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        For PyTorch backend, we bypass raw tensor inference and use
        Ultralytics predict() for correctness. The caller should use
        `predict_pytorch()` instead.
        """
        raise NotImplementedError(
            "Use predict_pytorch() for PyTorch backend"
        )

    @property
    def yolo(self):
        return self._yolo

    @property
    def class_names(self):
        return self._class_names


class Detector:
    """
    Unified road anomaly detector with multi-backend support.

    Usage
    -----
    >>> from pi_optimized.config import get_default_config
    >>> cfg = get_default_config()
    >>> detector = Detector(cfg)
    >>> result = detector.detect(cv2.imread("test.jpg"))
    >>> print(result.to_json())
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or get_default_config()
        self.cfg.ensure_directories()

        self._backend: Optional[BaseBackend] = None
        self._initialized = False

    def initialize(self) -> None:
        """Load model and warm up the inference backend."""
        if self._initialized:
            return

        model_path = str(self.cfg.resolve_model_path())
        logger.info(
            f"Initializing detector: backend={self.cfg.backend}, "
            f"model={model_path}"
        )

        if self.cfg.backend == "onnx":
            self._backend = ONNXBackend(model_path, self.cfg.num_threads)
        elif self.cfg.backend == "ncnn":
            self._backend = NCNNBackend(model_path, self.cfg.num_threads)
        elif self.cfg.backend == "pytorch":
            self._backend = PyTorchBackend(model_path, self.cfg.num_threads)
        else:
            raise ValueError(f"Unknown backend: {self.cfg.backend}")

        # Warm-up inference
        if self.cfg.backend != "pytorch":
            w, h = self.cfg.input_size
            logger.info("Running warm-up inference...")
            self._backend.warmup((1, 3, h, w))
            logger.info("Warm-up complete.")

        self._initialized = True

    def detect(
        self,
        img: np.ndarray,
        conf_threshold: Optional[float] = None,
        frame_index: int = 0,
    ) -> DetectionResult:
        """
        Run full detection pipeline on a single BGR image.

        Parameters
        ----------
        img : np.ndarray
            Input BGR image.
        conf_threshold : float, optional
            Override config confidence threshold.
        frame_index : int
            Frame counter (for video processing).

        Returns
        -------
        DetectionResult
            All detections with timing information.
        """
        if not self._initialized:
            self.initialize()

        conf = conf_threshold or self.cfg.conf_threshold

        # -------- Preprocess --------
        t0 = time.perf_counter()
        prep = preprocess(
            img,
            target_size=self.cfg.input_size,
            roi_crop=self.cfg.roi_crop,
            roi_top_fraction=self.cfg.roi_top_fraction,
        )
        t_preprocess = (time.perf_counter() - t0) * 1000

        # -------- Inference --------
        if self.cfg.backend == "pytorch":
            return self._detect_pytorch(img, conf, frame_index, t_preprocess)

        t1 = time.perf_counter()
        raw_output = self._backend.run(prep.tensor)
        t_inference = (time.perf_counter() - t1) * 1000

        # -------- Postprocess --------
        t2 = time.perf_counter()
        boxes, scores, class_ids = parse_yolov8_output(
            raw_output,
            conf_threshold=conf,
            iou_threshold=self.cfg.iou_threshold,
            max_detections=self.cfg.max_detections,
            num_classes=self.cfg.num_classes,
        )

        # Rescale boxes to original image coordinates
        boxes = rescale_detections(boxes, prep, self.cfg.input_size)
        t_postprocess = (time.perf_counter() - t2) * 1000

        # Build detection list
        detections = []
        for i in range(len(boxes)):
            cls_id = int(class_ids[i])
            cls_name = self.cfg.class_names.get(cls_id, f"class_{cls_id}")
            severity = self.cfg.severity_map.get(cls_name, "info")
            detections.append(
                Detection(
                    bbox=tuple(boxes[i].astype(int).tolist()),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=float(scores[i]),
                    severity=severity,
                )
            )

        return DetectionResult(
            detections=detections,
            inference_ms=t_inference,
            preprocess_ms=t_preprocess,
            postprocess_ms=t_postprocess,
            frame_index=frame_index,
        )

    def _detect_pytorch(
        self,
        img: np.ndarray,
        conf: float,
        frame_index: int,
        t_preprocess: float,
    ) -> DetectionResult:
        """PyTorch/Ultralytics path — uses native predict() for reliability."""
        backend: PyTorchBackend = self._backend

        t1 = time.perf_counter()
        results = backend.yolo.predict(
            img,
            conf=conf,
            iou=self.cfg.iou_threshold,
            max_det=self.cfg.max_detections,
            imgsz=self.cfg.input_size[0],
            verbose=False,
        )[0]
        t_inference = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        detections = []
        if results.boxes is not None and len(results.boxes):
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            names = backend.class_names
            for i in range(len(boxes_xyxy)):
                cls_id = int(cls_ids[i])
                cls_name = names.get(cls_id, f"class_{cls_id}")
                severity = self.cfg.severity_map.get(cls_name, "info")
                detections.append(
                    Detection(
                        bbox=tuple(boxes_xyxy[i].astype(int).tolist()),
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=float(confs[i]),
                        severity=severity,
                    )
                )
        t_postprocess = (time.perf_counter() - t2) * 1000

        return DetectionResult(
            detections=detections,
            inference_ms=t_inference,
            preprocess_ms=t_preprocess,
            postprocess_ms=t_postprocess,
            frame_index=frame_index,
        )

    def annotate(
        self,
        img: np.ndarray,
        result: DetectionResult,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draw detections and stats on image.

        Parameters
        ----------
        img : np.ndarray
            Original BGR image.
        result : DetectionResult
            Detection results.
        fps : float, optional
            Current FPS to display.

        Returns
        -------
        np.ndarray
            Annotated image.
        """
        annotated = draw_detections(
            img.copy(),
            result.detections,
            self.cfg.class_colors,
            thickness=2,
            font_scale=0.5,
        )
        if self.cfg.enable_profiling or fps is not None:
            annotated = draw_stats_overlay(annotated, result, fps)
        return annotated

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return (
            f"Detector(backend={self.cfg.backend}, "
            f"input_size={self.cfg.input_size}, "
            f"conf={self.cfg.conf_threshold}, status={status})"
        )
