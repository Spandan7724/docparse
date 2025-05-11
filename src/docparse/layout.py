# src/docparse/layout.py

import numpy as np
import onnxruntime as ort
import cv2
from typing import List, Dict
from pdf_backend_pdfium import render_page

# ——— Rasteriser façade ————————————————————————————————

def raster_page(path: str, page_idx: int, dpi: int = 224) -> np.ndarray:
    """
    Use the Rust extension’s render_page which returns (w, h, raw_bytes).
    Convert raw bytes → H×W×3 uint8 RGB NumPy array.
    """
    w, h, raw = render_page(path, page_idx, dpi)
    # raw is a python bytes object; buffer-friendly
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr.reshape((h, w, 3))


# ——— ONNX detector façade ————————————————————————————

_session: ort.InferenceSession = None

def get_session(model_path: str = "models/yolov8s-doclaynet.onnx") -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    return _session

def preprocess(
    image: np.ndarray,
    input_size: int = 1024
) -> np.ndarray:
    """
    Resize & pad to square, normalize to [0,1], and NCHW.
    """
    h, w, _ = image.shape
    scale = input_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(image, (nw, nh))
    pad_h, pad_w = input_size - nh, input_size - nw
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                              cv2.BORDER_CONSTANT, value=(114,114,114))
    # BGR→RGB if your model wants it:
    img = img[..., ::-1]
    img = img.astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None, ...]

def postprocess(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    orig_shape: tuple[int, int],
    input_size: int = 1024,
    score_thresh: float = 0.3,
    iou_thresh: float = 0.45,
    class_map: Dict[int,str] = None
) -> List[Dict]:
    """
    Filter, NMS & scale boxes back to original image.
    """
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
    model_path: str = "layout.onnx",
    class_map: Dict[int,str] = None
) -> List[Dict]:
    """
    Run your ONNX layout model on the image and return a list of regions.
    """
    sess = get_session(model_path)
    inp = preprocess(image)  # shape (1, C, H, W)
    outputs = sess.run(None, {"images": inp})
    dets = outputs[0]
    preds = dets[0]
    boxes  = preds[:, 0:4]  # x1, y1, x2, y2
    scores = preds[:, 4]
    labels = preds[:, 5]
    return postprocess(
        boxes, scores, labels,
        image.shape[:2],  # original shape
        class_map=class_map
    )