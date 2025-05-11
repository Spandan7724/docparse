# src/docparse/layout.py

import numpy as np
import cv2
from pathlib import Path
from pdf_backend_pdfium import render_page

def raster_page(path: str, page_idx: int, dpi: int = 224) -> np.ndarray:
    """
    Returns an H×W×3 uint8 RGB image from the given PDF page.
    """
    w, h, raw = render_page(path, page_idx, dpi)
    arr = np.frombuffer(raw, dtype=np.uint8)
    arr = arr.reshape((h, w, 3))
    return arr

def detect_regions(image: np.ndarray, model_path: str = "layout.onnx"):
    import onnxruntime as ort
    # ... your existing detect_regions code, now using raster_page ...
    pass
