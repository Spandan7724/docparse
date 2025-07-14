# src/docparse/layout.py

import numpy as np
import onnxruntime as ort
import cv2
from typing import List, Dict, Optional
from pdf_backend_pdfium import render_page
from .config import get_config

# ——— Rasteriser façade ————————————————————————————————

def raster_page(path: str, page_idx: int, dpi: Optional[int] = None) -> np.ndarray:
    """
    Use the Rust extension’s render_page which returns (w, h, raw_bytes).
    Convert raw bytes → H×W×3 uint8 RGB NumPy array.
    """
    if dpi is None:
        dpi = get_config().layout_detection.dpi
    w, h, raw = render_page(path, page_idx, dpi)
    # raw is a python bytes object; buffer-friendly
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr.reshape((h, w, 3))


# ——— ONNX detector façade ————————————————————————————

_session: ort.InferenceSession = None

def get_session(model_path: Optional[str] = None) -> ort.InferenceSession:
    global _session
    if _session is None:
        if model_path is None:
            config = get_config()
            model_path = str(config.get_model_path())
            providers = config.runtime.execution_providers
        else:
            providers = get_config().runtime.execution_providers
        
        _session = ort.InferenceSession(
            str(model_path),
            providers=providers
        )
    return _session

def preprocess(
    image: np.ndarray,
    input_size: Optional[int] = None
) -> np.ndarray:
    """
    Resize & pad to square, normalize to [0,1], and NCHW.
    """
    if input_size is None:
        input_size = get_config().layout_detection.input_size
    
    config = get_config()
    padding_color = tuple(config.layout_detection.padding_color)
    
    h, w, _ = image.shape
    scale = input_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(image, (nw, nh))
    pad_h, pad_w = input_size - nh, input_size - nw
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                              cv2.BORDER_CONSTANT, value=padding_color)
    # BGR→RGB if your model wants it:
    img = img[..., ::-1]
    img = img.astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None, ...]

def postprocess(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    orig_shape: tuple[int, int],
    input_size: Optional[int] = None,
    score_thresh: Optional[float] = None,
    iou_thresh: Optional[float] = None,
    class_map: Optional[Dict[int,str]] = None
) -> List[Dict]:
    """
    Filter, NMS & scale boxes back to original image.
    """
    config = get_config()
    if input_size is None:
        input_size = config.layout_detection.input_size
    if score_thresh is None:
        score_thresh = config.layout_detection.confidence_threshold
    if iou_thresh is None:
        iou_thresh = config.layout_detection.iou_threshold
    
    # 1) score filter
    mask = scores > score_thresh
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
    if boxes.size == 0:
        return []

    # 2) NMS
    bboxes = boxes.tolist()
    idxs = cv2.dnn.NMSBoxes(bboxes, scores.tolist(), score_thresh, iou_thresh)

    # flatten out any array/list-of-lists to a simple List[int]
    final_idxs: List[int] = []
    if isinstance(idxs, np.ndarray):
        final_idxs = idxs.flatten().tolist()
    elif isinstance(idxs, (list, tuple)):
        for x in idxs:
            if isinstance(x, (list, tuple, np.ndarray)):
                final_idxs.append(int(x[0]))
            else:
                final_idxs.append(int(x))

    H0, W0 = orig_shape
    gain = max(H0, W0) / input_size
    results: List[Dict] = []
    for i in final_idxs:
        x1, y1, x2, y2 = boxes[i]
        # undo scale & pad
        x1, x2 = x1 * gain, x2 * gain
        y1, y2 = y1 * gain, y2 * gain

        results.append({
            "bbox": [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1)
            ],
            "score": float(scores[i]),
            "label": class_map[int(labels[i])] if class_map else int(labels[i])
        })
    return results



def detect_regions(
    image: np.ndarray,
    model_path: Optional[str] = None,
    class_map: Optional[Dict[int,str]] = None,
    confidence_threshold: Optional[float] = None,
    input_size: Optional[int] = None,
    iou_threshold: Optional[float] = None
) -> List[Dict]:
    """
    Run your ONNX layout model on the image and return a list of regions.
    """
    sess = get_session(model_path)
    inp = preprocess(image, input_size=input_size)  # shape (1, C, H, W)
    outputs = sess.run(None, {"images": inp})
    dets = outputs[0]
    preds = dets[0]
    boxes  = preds[:, 0:4]  # x1, y1, x2, y2
    scores = preds[:, 4]
    labels = preds[:, 5]
    return postprocess(
        boxes, scores, labels,
        image.shape[:2],  # original shape
        input_size=input_size,
        score_thresh=confidence_threshold,
        iou_thresh=iou_threshold,
        class_map=class_map
    )