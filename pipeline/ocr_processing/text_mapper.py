import math
import numpy as np


def _box_area(b):
    """Calculate area of bbox (x1, y1, x2, y2)"""
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _intersection_area(a, b):
    """Calculate intersection area of two bboxes"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def get_text_rotation_for_shape(shape_bbox, ocr_data, threshold=0.25):
    """
    Find OCR text within a shape and calculate weighted rotation angle.
    
    Args:
        shape_bbox: (x1, y1, x2, y2) of the shape
        ocr_data: List of OCR data with polygon_rotation/shape_rotation
        threshold: Minimum overlap threshold (0-1)
    
    Returns:
        Snapped rotation angle (float) or None if no text found
    """
    sx1, sy1, sx2, sy2 = shape_bbox

    weighted_vector = np.array([0.0, 0.0])
    total_weight = 0
    matched = False

    for item in ocr_data:
        tx1, ty1, tx2, ty2 = item["bbox"]
        text_area = _box_area(item["bbox"])
        if text_area == 0:
            continue

        inter = _intersection_area(shape_bbox, item["bbox"])
        overlap = inter / text_area
        if overlap < threshold:
            continue

        matched = True
        conf = float(item.get("confidence", 1.0))

        # Use polygon_rotation if available, fallback to shape_rotation
        ang = item.get("polygon_rotation", item.get("shape_rotation", 0))
        if ang is None:
            ang = 0

        rad = math.radians(ang)
        ux, uy = math.cos(rad), math.sin(rad)

        w = overlap * conf * max(1, text_area)
        weighted_vector += w * np.array([ux, uy])
        total_weight += w

    if not matched or total_weight == 0:
        return None

    mean_vec = weighted_vector / total_weight
    mean_angle = (math.degrees(math.atan2(mean_vec[1], mean_vec[0])) + 360) % 360
    
    # Snap to standard angles
    STANDARD_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
    snapped = min(STANDARD_ANGLES, key=lambda s: min(abs(mean_angle - s), 360 - abs(mean_angle - s)))
    
    return float(snapped)


def map_ocr_to_nodes(diagram_nodes, processed_ocr_data, relaxation_pixels=15, max_area_ratio=80):
    nodes = [n.copy() for n in diagram_nodes]
    ocr_boxes = []

    if not processed_ocr_data:
        return nodes, []

    for ocr in processed_ocr_data:
        bbox = ocr.get('bbox')
        text = ocr.get('text', "")
        if not bbox or not text.strip():
            continue
        ocr_x1, ocr_y1, ocr_x2, ocr_y2 = bbox
        ocr_w = max(1, ocr_x2 - ocr_x1)
        ocr_h = max(1, ocr_y2 - ocr_y1)
        ocr_area = ocr_w * ocr_h
        mapped = False

        candidates = []
        for idx, node in enumerate(nodes):
            cx, cy, w, h = node['x'], node['y'], node['width'], node['height']
            node_x1 = cx - (w // 2) - relaxation_pixels
            node_y1 = cy - (h // 2) - relaxation_pixels
            node_x2 = cx + (w // 2) + relaxation_pixels
            node_y2 = cy + (h // 2) + relaxation_pixels
            if (ocr_x1 >= node_x1 and ocr_y1 >= node_y1 and ocr_x2 <= node_x2 and ocr_y2 <= node_y2):
                node_area = max(1, w * h)
                area_ratio = node_area / float(ocr_area)
                dx = (cx - (ocr_x1 + ocr_x2) / 2.0)
                dy = (cy - (ocr_y1 + ocr_y2) / 2.0)
                center_dist2 = dx * dx + dy * dy
                candidates.append((idx, node_area, area_ratio, center_dist2))

        if candidates:
            filtered = [c for c in candidates if c[2] <= max_area_ratio]
            chosen = min(filtered if filtered else candidates, key=lambda t: (t[1], t[3]))
            best_idx = chosen[0]
            node = nodes[best_idx]
            node['text'] = (node['text'] + " " + text).strip() if node['text'] else text
            mapped = True

        if not mapped:
            ocr_boxes.append({
                "text": text,
                "x1": int(ocr_x1), "y1": int(ocr_y1),
                "x2": int(ocr_x2), "y2": int(ocr_y2)
            })

    standalone_text_labels = []
    merge_y_thresh = 18
    merge_x_thresh = 40
    if ocr_boxes:
        ocr_boxes = sorted(ocr_boxes, key=lambda b: (b['y1'], b['x1']))
        used = [False] * len(ocr_boxes)
        for i, box in enumerate(ocr_boxes):
            if used[i]:
                continue
            texts = [box['text'].strip()]
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            used[i] = True
            for j in range(i + 1, len(ocr_boxes)):
                if used[j]:
                    continue
                ob = ocr_boxes[j]
                ob_x1, ob_y1, ob_x2, ob_y2 = ob['x1'], ob['y1'], ob['x2'], ob['y2']
                vertical_close = abs(ob_y1 - y2) <= merge_y_thresh or abs(ob_y2 - y1) <= merge_y_thresh
                horizontal_overlap = (ob_x1 <= x2 + merge_x_thresh and ob_x2 >= x1 - merge_x_thresh)
                if vertical_close and horizontal_overlap:
                    texts.append(ob['text'].strip())
                    x1 = min(x1, ob_x1)
                    y1 = min(y1, ob_y1)
                    x2 = max(x2, ob_x2)
                    y2 = max(y2, ob_y2)
                    used[j] = True
            standalone_text_labels.append({
                "id": f"text{len(standalone_text_labels) + 1}",
                "x": int(round((x1 + x2) / 2)),
                "y": int(round((y1 + y2) / 2)),
                "text": " ".join(texts).strip(),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "width": x2 - x1,
                "height": y2 - y1
            })

    return nodes, standalone_text_labels
