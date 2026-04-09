#!/usr/bin/env python3
"""
Model Export — YOLOv8 .pt → ONNX (with optional quantization)
===============================================================
Exports the existing trained YOLOv8m best.pt to ONNX format
optimized for Raspberry Pi CPU inference.

No retraining required — uses the existing weights directly.

Usage
-----
    python -m pi_optimized.export.export_model
    python -m pi_optimized.export.export_model --input best.pt --output model.onnx --imgsz 320
    python -m pi_optimized.export.export_model --quantize int8
"""

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))


def export_to_onnx(
    input_path: str,
    output_path: str,
    imgsz: int = 320,
    opset: int = 17,
    simplify: bool = True,
    half: bool = False,
    dynamic: bool = False,
) -> Path:
    """
    Export YOLOv8 .pt model to ONNX format.

    Parameters
    ----------
    input_path : str
        Path to .pt weights file.
    output_path : str
        Desired output .onnx path.
    imgsz : int
        Model input resolution.
    opset : int
        ONNX opset version.
    simplify : bool
        Whether to simplify the ONNX graph.
    half : bool
        Export in FP16 (not recommended for CPU inference).
    dynamic : bool
        Use dynamic input shapes.

    Returns
    -------
    Path
        Path to exported .onnx file.
    """
    from ultralytics import YOLO

    logger.info(f"Loading model from: {input_path}")
    model = YOLO(input_path)

    logger.info(
        f"Exporting to ONNX: imgsz={imgsz}, opset={opset}, "
        f"simplify={simplify}, half={half}"
    )

    # Ultralytics export handles the ONNX conversion
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        half=half,
        dynamic=dynamic,
    )

    export_path = Path(export_path)

    # Move to desired output location
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if export_path != output:
        import shutil
        shutil.move(str(export_path), str(output))
        logger.info(f"Moved exported model to: {output}")

    # Report file size
    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info(f"Export complete: {output} ({size_mb:.1f} MB)")

    return output


def quantize_onnx(
    input_path: str,
    output_path: str,
    quant_type: str = "dynamic",
) -> Path:
    """
    Quantize ONNX model for reduced size and faster CPU inference.

    Parameters
    ----------
    input_path : str
        Path to FP32 .onnx model.
    output_path : str
        Output path for quantized model.
    quant_type : str
        'dynamic' (recommended for CPU) or 'static'.

    Returns
    -------
    Path
        Path to quantized model.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.error(
            "onnxruntime quantization tools not available. "
            "Install with: pip install onnxruntime"
        )
        raise

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Quantizing model: {quant_type} quantization")

    if quant_type == "dynamic":
        quantize_dynamic(
            model_input=input_path,
            model_output=str(output),
            weight_type=QuantType.QUInt8,
            optimize_model=True,
        )
    else:
        logger.warning(
            "Static quantization requires calibration data. "
            "Using dynamic quantization as fallback."
        )
        quantize_dynamic(
            model_input=input_path,
            model_output=str(output),
            weight_type=QuantType.QUInt8,
        )

    orig_size = Path(input_path).stat().st_size / (1024 * 1024)
    quant_size = output.stat().st_size / (1024 * 1024)
    compression = (1 - quant_size / orig_size) * 100

    logger.info(
        f"Quantization complete: {orig_size:.1f} MB → {quant_size:.1f} MB "
        f"({compression:.1f}% reduction)"
    )

    return output


def validate_onnx(onnx_path: str, test_shape: tuple = (1, 3, 320, 320)) -> bool:
    """
    Validate that the ONNX model loads and runs correctly.

    Parameters
    ----------
    onnx_path : str
        Path to .onnx model.
    test_shape : tuple
        Input tensor shape for test inference.

    Returns
    -------
    bool
        True if validation passes.
    """
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed for validation.")
        return False

    logger.info(f"Validating ONNX model: {onnx_path}")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_names = [o.name for o in session.get_outputs()]

    logger.info(f"  Input: {input_name} shape={input_shape}")
    logger.info(f"  Outputs: {output_names}")

    # Test inference
    dummy = np.random.randn(*test_shape).astype(np.float32)
    outputs = session.run(output_names, {input_name: dummy})

    logger.info(f"  Output shape: {outputs[0].shape}")
    logger.info(f"  Output dtype: {outputs[0].dtype}")
    logger.info("  ✅ Validation PASSED — model loads and runs successfully")

    return True


def export_to_ncnn(
    input_path: str,
    output_dir: str,
    imgsz: int = 320,
) -> Path:
    """
    Export YOLOv8 .pt model to NCNN format.

    Parameters
    ----------
    input_path : str
        Path to .pt model.
    output_dir : str
        Directory for NCNN output files (.param + .bin).
    imgsz : int
        Model input resolution.

    Returns
    -------
    Path
        Output directory containing model files.
    """
    from ultralytics import YOLO

    logger.info(f"Exporting to NCNN: {input_path}")
    model = YOLO(input_path)

    export_path = model.export(
        format="ncnn",
        imgsz=imgsz,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    export_path = Path(export_path)
    if export_path.is_dir() and export_path != output:
        import shutil
        # Copy NCNN files to target directory
        for f in export_path.glob("*"):
            shutil.copy2(str(f), str(output / f.name))
        logger.info(f"NCNN model files copied to: {output}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to ONNX/NCNN for Pi deployment"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(
            _REPO_ROOT / "RoadDetectionModel"
            / "RoadModel_yolov8m.pt_rounds120_b9"
            / "weights" / "best.pt"
        ),
        help="Path to .pt model file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(_PROJECT_ROOT / "models" / "best_road.onnx"),
        help="Output model path",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["onnx", "ncnn"],
        default="onnx",
        help="Export format",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Model input resolution (default: 320)",
    )
    parser.add_argument(
        "--quantize", "-q",
        choices=["none", "dynamic", "int8"],
        default="none",
        help="Quantization type (ONNX only)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after export",
    )

    args = parser.parse_args()

    if args.format == "onnx":
        # Export to ONNX
        onnx_path = export_to_onnx(
            input_path=args.input,
            output_path=args.output,
            imgsz=args.imgsz,
        )

        # Optional quantization
        if args.quantize != "none":
            quant_output = str(onnx_path).replace(".onnx", f"_q{args.quantize}.onnx")
            quantize_onnx(
                input_path=str(onnx_path),
                output_path=quant_output,
                quant_type=args.quantize,
            )
            onnx_path = Path(quant_output)

        # Validation
        if args.validate:
            h = w = args.imgsz
            validate_onnx(str(onnx_path), test_shape=(1, 3, h, w))

    elif args.format == "ncnn":
        export_to_ncnn(
            input_path=args.input,
            output_dir=args.output,
            imgsz=args.imgsz,
        )

    logger.info("Export pipeline complete!")


if __name__ == "__main__":
    main()
