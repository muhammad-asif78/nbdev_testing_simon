from typing import List, Tuple, Dict, Any
import cv2
import os

from rf_detr.detector import detect_shapes
from paddle_ocr.ocr import extract_text

from .angle_detection import predict_angles, predict_angles_with_masks
from .arrow_processing import find_arrow_endpoints
from .shape_processing import attach_colors, establish_connections
from .ocr_processing import map_ocr_to_nodes, get_text_rotation_for_shape
from .json_builders import create_nodes_json, attach_angles_to_nodes, create_edges_json

ANGLE_ALLOWED_SHAPES = {"rectangle", "Triangle", "Racetrack", "Pentagon", "arrow"}


def _ensure_image_from_pdf_if_needed(file_path: str) -> str:
    if file_path.lower().endswith('.pdf'):
        try:
            import fitz
            import tempfile
            doc = fitz.open(file_path)
            page = doc.load_page(0)
            pix = page.get_pixmap()
            with tempfile.NamedTemporaryFile(suffix="_page0.png", delete=False) as tmp:
                out_path = tmp.name
            pix.save(out_path)
            doc.close()
            return out_path
        except Exception:
            return file_path
    return file_path


def _get_ocr_based_angles(detections, ocr_results: List[Dict]) -> Dict[Any, str]:
    ocr_angle_map = {}
    
    if not ocr_results:
        return ocr_angle_map
    
    for det in detections:
        label = det[0]
        tl = det[1]
        br = det[2]
        
        if label not in ANGLE_ALLOWED_SHAPES:
            continue
            
        x1, y1 = int(round(tl[0])), int(round(tl[1]))
        x2, y2 = int(round(br[0])), int(round(br[1]))
        shape_bbox = (x1, y1, x2, y2)
        
        rotation = get_text_rotation_for_shape(shape_bbox, ocr_results)
        
        if rotation is not None:
            det_key = (det[0], det[1], det[2])
            ocr_angle_map[det_key] = f"{int(rotation)}"
    
    return ocr_angle_map


def run_spatial_mapping(file_path: str) -> Dict[str, Any]:
    img_path = _ensure_image_from_pdf_if_needed(file_path)
    detections = detect_shapes(img_path)
    ocr_results = extract_text(img_path)
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {img_path}")

    H, W = image_bgr.shape[:2]

    if not detections:
        nodes_json: List[Dict[str, Any]] = []
        standalone_text_labels = []
        if ocr_results:
            nodes_json, standalone_text_labels = map_ocr_to_nodes([], ocr_results)
        return {
            "canvas": {"width": int(W), "height": int(H)},
            "nodes": nodes_json,
            "edges": [],
            "text_labels": standalone_text_labels
        }

    enriched = attach_colors(image_bgr, detections)
    
    ocr_angle_map = _get_ocr_based_angles(detections, ocr_results)
    
    resnet_angle_map = predict_angles_with_masks(image_bgr, detections)
    if not resnet_angle_map:
        resnet_angle_map = predict_angles(image_bgr, detections)
    
    angle_map = {}
    for det in detections:
        det_key = (det[0], det[1], det[2])
        if det_key in ocr_angle_map:
            angle_map[det_key] = ocr_angle_map[det_key]
        elif det in resnet_angle_map:
            angle_map[det_key] = resnet_angle_map[det]
        elif det_key in resnet_angle_map:
            angle_map[det_key] = resnet_angle_map[det_key]

    vectors = find_arrow_endpoints(image_bgr, detections, ocr_results)
    connections = establish_connections(vectors, detections)

    shape_only = [d for d in enriched if d[0] not in ["dashed-arrow", "dotted-arrow", "solid-arrow", "arrow_head"]]
    
    # Sort shapes by area (largest first) for proper layering
    # Background/container shapes will appear before inner/nested shapes
    def _get_area(det):
        x1, y1 = det[1]
        x2, y2 = det[2]
        return (x2 - x1) * (y2 - y1)
    
    shape_only_sorted = sorted(shape_only, key=_get_area, reverse=True)
    nodes_json = create_nodes_json(shape_only_sorted)
    nodes_json = attach_angles_to_nodes(nodes_json, detections, angle_map)
    
    standalone_text_labels = []  
    if ocr_results:
        nodes_json, standalone_text_labels = map_ocr_to_nodes(nodes_json, ocr_results)

    edges_json = create_edges_json(connections, nodes_json, enriched)

    for node in nodes_json:
        node["x"] = node["x"] - node["width"] // 2
        node["y"] = node["y"] - node["height"] // 2

    final_json_output = {
        "canvas": {"width": int(W), "height": int(H)},
        "nodes": nodes_json,
        "edges": edges_json,
        "text_labels": standalone_text_labels
    }
    
    return final_json_output
