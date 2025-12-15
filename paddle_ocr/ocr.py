from typing import List, Dict, Tuple, Any

import os
import json
import tempfile
import math

import cv2
import numpy as np


_PADDLE_OCR = None

# Standard angles for snapping
STANDARD_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
SNAP_TOLERANCE = 20


def _load_ocr():  
    global _PADDLE_OCR  
    if _PADDLE_OCR is not None:  
        return _PADDLE_OCR  
    try:  
        from paddleocr import PaddleOCR  
        _PADDLE_OCR = PaddleOCR(  
            text_detection_model_dir="./weights/paddleocr/PP-OCRv5_server_det_infer",  
            text_recognition_model_dir="./weights/paddleocr/PP-OCRv5_server_rec_infer",  
            textline_orientation_model_dir="weights/paddleocr/PP-LCNet_x1_0_textline_ori_infer",
            use_doc_orientation_classify=False,  
            use_doc_unwarping=False,  
            use_textline_orientation=True,  # Enabled for orientation extraction
            lang='en'  
        )  
        return _PADDLE_OCR  
    except Exception as exc:    
        print(f"[PaddleOCR] Failed to initialize: {exc}")  
        _PADDLE_OCR = None  
        return None


def _flatten_poly(polygon) -> np.ndarray:
    """Flatten nested polygon into Nx2 array"""
    arr = np.array(polygon, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    elif arr.ndim > 2:
        arr = arr.reshape(-1, 2)
    return arr


def calculate_rotation_from_polygon(polygon) -> float:
    """Calculate rotation angle from text polygon using PCA/SVD."""
    pts = _flatten_poly(polygon)
    n = pts.shape[0]

    if n < 2:
        return 0.0

    if n == 2:
        dx = pts[1, 0] - pts[0, 0]
        dy = pts[1, 1] - pts[0, 1]
        return (math.degrees(math.atan2(-dy, dx)) + 360) % 360

    centered = pts - pts.mean(axis=0)

    if np.allclose(centered, 0, atol=1e-6):
        return 0.0

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    vx, vy = Vt[0]
    pca_angle = (math.degrees(math.atan2(-vy, vx)) + 360) % 360

    x = pts[:, 0]
    y = pts[:, 1]
    if len(np.unique(x)) > 1:
        a, b = np.polyfit(x, y, 1)
        lf_angle = (math.degrees(math.atan2(-a, 1)) + 360) % 360
    else:
        lf_angle = pca_angle

    pca_var = np.var(centered @ np.array([vx, vy]))
    lf_dir = np.array([1, a]) / np.linalg.norm([1, a]) if len(np.unique(x)) > 1 else np.array([1, 0])
    lf_var = np.var(centered @ lf_dir)

    best = pca_angle if pca_var >= lf_var else lf_angle
    return round(best, 2)


def snap_to_standard_angle(angle: float, tolerance: int = SNAP_TOLERANCE) -> float:
    """Snap angle to nearest standard angle if within tolerance."""
    angle = angle % 360
    best = min(STANDARD_ANGLES, key=lambda s: abs(angle - s))
    return float(best) if abs(best - angle) <= tolerance else round(angle, 1)


def _preprocess_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 3, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image_bgr, -1, kernel)
    alpha = 0.5
    blended = cv2.addWeighted(image_bgr, 1 - alpha, sharpened, alpha, 0)
    return blended


def extract_text(file_path: str) -> List[Dict[str, Tuple[int, int, int, int]]]:

    """ Run PaddleOCR and return list of {text, confidence, bbox(x1,y1,x2,y2)}  """

    # Set confidence threshold to filter low-confidence OCR results
    min_confidence = 0.4

    ocr = _load_ocr()
    if ocr is None:
        return []

    if not os.path.exists(file_path):
        return []

    image_bgr = cv2.imread(file_path)
    if image_bgr is None:
        return []

    image_bgr = _preprocess_for_ocr(image_bgr)

    try:
        result = ocr.predict(image_bgr)
        outputs: List[Dict] = []
        if not result:
            return outputs
        # Persist first result to JSON for stable parsing
        tmp_dir = tempfile.gettempdir()
        tmp_json = os.path.join(tmp_dir, "paddle_ocr_result.json")
        try:
            first = result[0]
            # Some versions expose .save_to_json, others not
            if hasattr(first, "save_to_json"):
                first.save_to_json(tmp_json)
            else:
                # Try to serialize via .res
                if hasattr(first, "res") and isinstance(first.res, dict):
                    with open(tmp_json, "w", encoding="utf-8") as f:
                        json.dump(first.res, f)
                else:
                    
                    tmp_json = None
        except Exception:
            tmp_json = None

        if tmp_json and os.path.exists(tmp_json):
            try:
                with open(tmp_json, "r", encoding="utf-8") as f:
                    ocr_json = json.load(f)
                rec_texts = ocr_json.get("rec_texts", [])
                rec_scores = ocr_json.get("rec_scores", [])
                rec_polys = ocr_json.get("rec_polys", [])
                orientation_angles = ocr_json.get("textline_orientation_angles", [])
                
                for i, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
                    # Apply confidence threshold filter
                    if float(score) < min_confidence:
                        continue
                    pts = _flatten_poly(poly)
                    x1, y1 = np.min(pts, axis=0)
                    x2, y2 = np.max(pts, axis=0)
                    
                    # Calculate rotation from polygon
                    raw_rotation = calculate_rotation_from_polygon(poly)
                    snapped_rotation = snap_to_standard_angle(raw_rotation)
                    
                    outputs.append({
                        "text": str(text),
                        "confidence": float(score),
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "polygon": poly,
                        "paddleocr_orientation": orientation_angles[i] if i < len(orientation_angles) else None,
                        "polygon_rotation": raw_rotation,
                        "shape_rotation": snapped_rotation
                    })
                return outputs
            except Exception:
                
                pass

        
        try:
            first = result[0]
            rec_texts = first.res.get("rec_texts") if hasattr(first, "res") else None
            rec_scores = first.res.get("rec_scores") if hasattr(first, "res") else None
            rec_polys = first.res.get("rec_polys") if hasattr(first, "res") else None
            orientation_angles = first.res.get("textline_orientation_angles", []) if hasattr(first, "res") else []
            if rec_texts is None or rec_scores is None or rec_polys is None:
                return outputs
            for i, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
                # Apply confidence threshold filter
                if float(score) < min_confidence:
                    continue
                pts = _flatten_poly(poly)
                x1, y1 = np.min(pts, axis=0)
                x2, y2 = np.max(pts, axis=0)
                
                # Calculate rotation from polygon
                raw_rotation = calculate_rotation_from_polygon(poly)
                snapped_rotation = snap_to_standard_angle(raw_rotation)
                
                outputs.append({
                    "text": str(text),
                    "confidence": float(score),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "polygon": poly,
                    "paddleocr_orientation": orientation_angles[i] if i < len(orientation_angles) else None,
                    "polygon_rotation": raw_rotation,
                    "shape_rotation": snapped_rotation
                })
            return outputs
        except Exception:
            return outputs
    except Exception as exc:  
        print(f"[PaddleOCR] Inference failed: {exc}")
        return []
