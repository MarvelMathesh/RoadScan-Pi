"""
Post-processing Pipeline
=========================
NMS, detection dataclass, lightweight annotation (pure OpenCV).
Replaces the heavy `supervision` library dependency.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class Detection:
    """Single detection result."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in image coords
    class_id: int
    class_name: str
    confidence: float
    severity: str = "info"  # 'info' | 'warning' | 'critical'

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
        }


@dataclass
class DetectionResult:
    """Collection of detections for a single frame."""

    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    frame_index: int = 0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    @property
    def count(self) -> int:
        return len(self.detections)

    def by_severity(self, severity: str) -> list[Detection]:
        return [d for d in self.detections if d.severity == severity]

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "count": self.count,
            "inference_ms": round(self.inference_ms, 2),
            "preprocess_ms": round(self.preprocess_ms, 2),
            "postprocess_ms": round(self.postprocess_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "frame_index": self.frame_index,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """
    Non-Maximum Suppression implemented in pure NumPy.
    Avoids dependency on torch.ops.nms or cv2.dnn.NMSBoxes quirks.

    Parameters
    ----------
    boxes : np.ndarray
        (N, 4) array of [x1, y1, x2, y2].
    scores : np.ndarray
        (N,) confidence scores.
    iou_threshold : float
        IoU threshold for suppression.

    Returns
    -------
    np.ndarray
        Indices of kept detections.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # Compute IoU of the kept box with remaining
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union_area = areas[i] + areas[order[1:]] - inter_area
        iou = inter_area / np.maximum(union_area, 1e-6)

        # Keep boxes with IoU below threshold
        mask = iou <= iou_threshold
        order = order[1:][mask]

    return np.array(keep, dtype=np.int32)


def parse_yolov8_output(
    output: np.ndarray,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    max_detections: int = 50,
    num_classes: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse raw YOLOv8 output tensor into detections.

    YOLOv8 output shape: (1, 4 + num_classes, num_proposals)
    where proposals are in xywh format.

    Parameters
    ----------
    output : np.ndarray
        Raw model output, shape (1, 4+C, N) or (4+C, N).
    conf_threshold : float
        Minimum confidence score.
    iou_threshold : float
        NMS IoU threshold.
    max_detections : int
        Maximum number of detections to return.
    num_classes : int
        Number of detection classes.

    Returns
    -------
    boxes : np.ndarray
        (K, 4) in xyxy format, model-input coordinates.
    scores : np.ndarray
        (K,) confidence scores.
    class_ids : np.ndarray
        (K,) class indices.
    """
    # Handle batch dimension
    if output.ndim == 3:
        output = output[0]  # (4+C, N)

    # Transpose if needed: some exports give (N, 4+C)
    if output.shape[0] > output.shape[1]:
        output = output.T  # → (4+C, N)

    # Split into boxes and class scores
    boxes_xywh = output[:4, :]   # (4, N)
    class_scores = output[4:, :]  # (C, N)

    # Max class confidence per proposal
    max_scores = class_scores.max(axis=0)       # (N,)
    class_ids = class_scores.argmax(axis=0)     # (N,)

    # Confidence filter
    mask = max_scores >= conf_threshold
    if not np.any(mask):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    filtered_boxes = boxes_xywh[:, mask].T       # (K, 4) xywh
    filtered_scores = max_scores[mask]            # (K,)
    filtered_class_ids = class_ids[mask]          # (K,)

    # Convert xywh → xyxy
    xyxy = np.empty_like(filtered_boxes)
    xyxy[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2  # x1
    xyxy[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2  # y1
    xyxy[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2  # x2
    xyxy[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2  # y2

    # Per-class NMS
    keep_indices = []
    for cls_id in range(num_classes):
        cls_mask = filtered_class_ids == cls_id
        if not np.any(cls_mask):
            continue
        cls_boxes = xyxy[cls_mask]
        cls_scores = filtered_scores[cls_mask]
        cls_keep = nms_numpy(cls_boxes, cls_scores, iou_threshold)

        # Map back to global indices
        global_indices = np.where(cls_mask)[0][cls_keep]
        keep_indices.extend(global_indices.tolist())

    if not keep_indices:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    keep_indices = np.array(keep_indices, dtype=np.int32)

    # Sort by confidence descending and limit
    sorted_order = filtered_scores[keep_indices].argsort()[::-1][:max_detections]
    keep_indices = keep_indices[sorted_order]

    return (
        xyxy[keep_indices],
        filtered_scores[keep_indices],
        filtered_class_ids[keep_indices].astype(np.int32),
    )


def draw_detections(
    img: np.ndarray,
    detections: list[Detection],
    class_colors: dict[int, tuple[int, int, int]],
    thickness: int = 2,
    font_scale: float = 0.5,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Draw detection boxes and labels on image using pure OpenCV.
    No dependency on supervision library.

    Parameters
    ----------
    img : np.ndarray
        BGR image to annotate (modified in-place).
    detections : list[Detection]
        Detection results.
    class_colors : dict
        Class ID → BGR colour mapping.
    thickness : int
        Box line thickness.
    font_scale : float
        Label text scale.
    show_confidence : bool
        Whether to show confidence in label.

    Returns
    -------
    np.ndarray
        Annotated image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = class_colors.get(det.class_id, (128, 128, 128))

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Build label
        label = det.class_name
        if show_confidence:
            label = f"{label} {det.confidence:.2f}"

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        label_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(
            img, (x1, label_y1), (x1 + tw + 4, y1), color, cv2.FILLED
        )

        # Label text — choose contrasting text colour
        brightness = (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299)
        txt_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - baseline - 2),
            font,
            font_scale,
            txt_color,
            1,
            cv2.LINE_AA,
        )

    return img


def draw_stats_overlay(
    img: np.ndarray,
    result: DetectionResult,
    fps: Optional[float] = None,
) -> np.ndarray:
    """
    Draw performance stats overlay in corner of frame.
    """
    lines = []
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    lines.append(f"Detections: {result.count}")
    lines.append(f"Inference: {result.inference_ms:.0f}ms")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (0, 255, 0)
    y_offset = 20

    for i, line in enumerate(lines):
        y = y_offset + i * 22
        # Shadow
        cv2.putText(img, line, (11, y + 1), font, scale, (0, 0, 0), 2, cv2.LINE_AA)
        # Text
        cv2.putText(img, line, (10, y), font, scale, color, 1, cv2.LINE_AA)

    return img
