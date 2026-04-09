"""
Preprocessing Pipeline — Optimized for ARM
============================================
Letterbox resize, ROI cropping, normalization.
Minimizes memory allocations; works in-place where possible.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessResult:
    """Container for preprocessed frame and inverse-transform metadata."""

    tensor: np.ndarray          # (1, 3, H, W) float32 — model input
    scale: float                # Scale factor applied
    pad_w: int                  # Horizontal padding (left)
    pad_h: int                  # Vertical padding (top)
    original_shape: tuple[int, int]  # (orig_h, orig_w)


def letterbox(
    img: np.ndarray,
    target_size: tuple[int, int] = (320, 320),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, int, int]:
    """
    Resize image with aspect-ratio preservation + padding.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image (H, W, 3).
    target_size : tuple
        (width, height) for model input.
    color : tuple
        Padding fill colour.

    Returns
    -------
    padded : np.ndarray
        Letterboxed image at target_size.
    scale : float
        Scale factor applied.
    pad_w, pad_h : int
        Padding offsets.
    """
    h, w = img.shape[:2]
    tw, th = target_size

    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize with area interpolation (better for downscaling)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Compute padding
    pad_w = (tw - new_w) // 2
    pad_h = (th - new_h) // 2

    # Create padded canvas
    padded = np.full((th, tw, 3), color, dtype=np.uint8)
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    return padded, scale, pad_w, pad_h


def crop_roi(
    img: np.ndarray,
    top_fraction: float = 0.35,
) -> tuple[np.ndarray, int]:
    """
    Crop to bottom portion of frame (road surface focus).

    Parameters
    ----------
    img : np.ndarray
        Full frame.
    top_fraction : float
        Fraction of image height to discard from top.

    Returns
    -------
    cropped : np.ndarray
        Cropped frame.
    y_offset : int
        Pixel offset from original top.
    """
    h = img.shape[0]
    y_start = int(h * top_fraction)
    return img[y_start:], y_start


def preprocess(
    img: np.ndarray,
    target_size: tuple[int, int] = (320, 320),
    roi_crop: bool = False,
    roi_top_fraction: float = 0.35,
) -> PreprocessResult:
    """
    Full preprocessing pipeline: optional ROI crop → letterbox → normalize → NCHW.

    Parameters
    ----------
    img : np.ndarray
        Input BGR frame from OpenCV.
    target_size : tuple
        Model input (width, height).
    roi_crop : bool
        Whether to crop upper portion of frame.
    roi_top_fraction : float
        Fraction of top to discard if ROI cropping.

    Returns
    -------
    PreprocessResult
        Contains the model-ready tensor and inverse-transform metadata.
    """
    original_shape = img.shape[:2]  # (H, W)

    # Optional ROI crop
    if roi_crop:
        img, _ = crop_roi(img, roi_top_fraction)

    # Letterbox resize
    padded, scale, pad_w, pad_h = letterbox(img, target_size)

    # BGR → RGB (in-place not possible with channel swap, but cheap)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] float32
    blob = rgb.astype(np.float32) * (1.0 / 255.0)

    # HWC → CHW → NCHW
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])

    return PreprocessResult(
        tensor=blob,
        scale=scale,
        pad_w=pad_w,
        pad_h=pad_h,
        original_shape=original_shape,
    )


def rescale_detections(
    boxes: np.ndarray,
    prep: PreprocessResult,
    target_size: tuple[int, int] = (320, 320),
) -> np.ndarray:
    """
    Map detection boxes from model-space back to original image coordinates.

    Parameters
    ----------
    boxes : np.ndarray
        (N, 4) array of [x1, y1, x2, y2] in model input space.
    prep : PreprocessResult
        Preprocessing metadata.
    target_size : tuple
        Model input resolution.

    Returns
    -------
    np.ndarray
        (N, 4) boxes in original image coordinates.
    """
    if len(boxes) == 0:
        return boxes

    boxes = boxes.copy().astype(np.float32)

    # Remove padding offset
    boxes[:, [0, 2]] -= prep.pad_w
    boxes[:, [1, 3]] -= prep.pad_h

    # Undo scale
    boxes /= prep.scale

    # Clip to original image bounds
    orig_h, orig_w = prep.original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    return boxes
