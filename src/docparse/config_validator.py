"""Configuration validation utilities for docparse."""

import os
from pathlib import Path
from typing import List, Optional, Union

from .config import DocparseConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_config(config: DocparseConfig) -> List[str]:
    """
    Validate a configuration object and return a list of warnings.
    Raises ConfigValidationError for critical errors.
    """
    warnings = []
    
    # Validate model path
    model_path = config.get_model_path()
    if not model_path.exists():
        if config.layout_detection.model_path == "models/yolov8s-doclaynet.onnx":
            # Default model missing
            warnings.append(
                f"Default model not found at {model_path}. "
                "Layout detection will fail unless you specify a valid model path."
            )
        else:
            # Custom model missing - this is an error
            raise ConfigValidationError(f"Model file not found: {model_path}")
    
    # Validate model is readable
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                f.read(1)  # Try to read first byte
        except (IOError, PermissionError) as e:
            raise ConfigValidationError(f"Cannot read model file {model_path}: {e}")
    
    # Validate PDFium library path
    pdfium_path = Path(config.paths.pdfium_library_path)
    if pdfium_path.is_dir():
        # Check for libpdfium.so in the directory
        expected_lib = pdfium_path / "libpdfium.so"
        if not expected_lib.exists():
            warnings.append(
                f"libpdfium.so not found in {pdfium_path}. "
                "PDF processing may fail if the library is not in the system path."
            )
    
    # Validate DPI ranges
    if not (72 <= config.layout_detection.dpi <= 600):
        warnings.append(
            f"DPI value {config.layout_detection.dpi} is outside recommended range 72-600. "
            "This may cause performance issues or poor quality."
        )
    
    if not (72 <= config.rendering.default_dpi <= 600):
        warnings.append(
            f"Default DPI {config.rendering.default_dpi} is outside recommended range 72-600."
        )
    
    # Validate confidence thresholds
    if not (0.1 <= config.layout_detection.confidence_threshold <= 0.9):
        warnings.append(
            f"Confidence threshold {config.layout_detection.confidence_threshold} "
            "is outside typical range 0.1-0.9. You may get too many false positives or miss detections."
        )
    
    # Validate IoU threshold
    if not (0.1 <= config.layout_detection.iou_threshold <= 0.8):
        warnings.append(
            f"IoU threshold {config.layout_detection.iou_threshold} "
            "is outside typical range 0.1-0.8. This may affect detection quality."
        )
    
    # Validate input size is power of 2 or common size
    input_size = config.layout_detection.input_size
    common_sizes = [256, 512, 640, 1024, 1280, 1536]
    if input_size not in common_sizes:
        warnings.append(
            f"Input size {input_size} is not a common model size. "
            f"Common sizes are: {common_sizes}"
        )
    
    # Validate text extraction parameters
    if config.text_extraction.line_clustering_tolerance < 0.05:
        warnings.append(
            "Very low line clustering tolerance may cause characters to be split across lines."
        )
    elif config.text_extraction.line_clustering_tolerance > 0.5:
        warnings.append(
            "High line clustering tolerance may cause separate lines to be merged."
        )
    
    # Validate space thresholds
    if config.text_extraction.space_threshold_width > 1.0:
        warnings.append(
            "Space width threshold > 1.0 may cause missing spaces between words."
        )
    
    # Validate cache directory
    if config.paths.cache_directory:
        cache_path = Path(config.paths.cache_directory)
        if not cache_path.exists():
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                warnings.append(f"Cannot create cache directory {cache_path}: {e}")
        elif not os.access(cache_path, os.W_OK):
            warnings.append(f"Cache directory {cache_path} is not writable.")
    
    # Validate execution providers
    valid_providers = {
        "CPUExecutionProvider",
        "CUDAExecutionProvider", 
        "OpenVINOExecutionProvider",
        "TensorrtExecutionProvider",
        "DmlExecutionProvider",  # DirectML for Windows
    }
    
    unknown_providers = set(config.runtime.execution_providers) - valid_providers
    if unknown_providers:
        warnings.append(
            f"Unknown execution providers: {list(unknown_providers)}. "
            f"Valid providers: {list(valid_providers)}"
        )
    
    # Warn if only GPU providers without CPU fallback
    if config.runtime.execution_providers and "CPUExecutionProvider" not in config.runtime.execution_providers:
        warnings.append(
            "No CPU execution provider specified. "
            "Consider adding 'CPUExecutionProvider' as fallback."
        )
    
    return warnings


def validate_environment() -> List[str]:
    """
    Validate the runtime environment for docparse.
    Returns a list of warnings about potential issues.
    """
    warnings = []
    
    # Check for ONNX runtime
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        
        if "CUDAExecutionProvider" not in providers:
            warnings.append(
                "CUDA execution provider not available. "
                "Install onnxruntime-gpu for GPU acceleration."
            )
    except ImportError:
        warnings.append(
            "onnxruntime not installed. Layout detection will not work. "
            "Install with: pip install 'docparse[cpu]' or 'docparse[gpu]'"
        )
    
    # Check for OpenCV
    try:
        import cv2
    except ImportError:
        raise ConfigValidationError(
            "OpenCV not installed. This is required for image processing. "
            "Install with: pip install opencv-python"
        )
    
    # Check for numpy
    try:
        import numpy
    except ImportError:
        raise ConfigValidationError(
            "NumPy not installed. This is required for array processing."
        )
    
    return warnings


def safe_init_config(
    config_path: Optional[Union[str, Path]] = None,
    profile: Optional[str] = None,
    strict: bool = False,
    **overrides
) -> DocparseConfig:
    """
    Safely initialize configuration with validation.
    
    Args:
        config_path: Path to configuration file
        profile: Profile name to apply
        strict: If True, raise exception on warnings
        **overrides: Additional configuration overrides
    
    Returns:
        Validated configuration object
        
    Raises:
        ConfigValidationError: On critical validation errors or warnings in strict mode
    """
    from .config import init_config
    
    # Initialize configuration
    config = init_config(config_path=config_path, profile=profile, **overrides)
    
    # Validate configuration
    warnings = validate_config(config)
    env_warnings = validate_environment()
    all_warnings = warnings + env_warnings
    
    # Handle warnings
    if all_warnings:
        if strict:
            raise ConfigValidationError(
                f"Configuration validation failed:\n" + 
                "\n".join(f"- {w}" for w in all_warnings)
            )
        else:
            # Print warnings to stderr
            import sys
            print("Configuration warnings:", file=sys.stderr)
            for warning in all_warnings:
                print(f"  Warning: {warning}", file=sys.stderr)
    
    return config