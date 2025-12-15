def _to_snake_case(label):
    """Convert label to snake_case format."""
    return label.replace(" ", "_").replace("-", "_").lower()


def create_nodes_json(shape_detections_with_color):
    nodes = []
    for i, det in enumerate(shape_detections_with_color):
        # Handle both old format (label, tl, br, color) and new format (label, tl, br, color, confidence)
        if len(det) >= 5:
            label, tl, br, color, confidence = det[0], det[1], det[2], det[3], det[4]
        elif len(det) == 4:
            label, tl, br, color = det
            confidence = None
        else:
            continue
            
        if label in {"dashed-arrow", "dotted-arrow", "solid-arrow", "arrow_head"}:
            continue
        x1, y1 = tl
        x2, y2 = br
        nodes.append({
            "id": f"node{i+1}",
            "x": int(round((x1 + x2) / 2)),
            "y": int(round((y1 + y2) / 2)),
            "text": "",
            "shape": _to_snake_case(label),
            "color": color,
            "angle": 0,
            "width": int(round(x2 - x1)),
            "height": int(round(y2 - y1)),
            "bbox": {
                "x1": int(round(x1)),
                "y1": int(round(y1)),
                "x2": int(round(x2)),
                "y2": int(round(y2))
            },
            "confidence": round(float(confidence), 4) if confidence is not None else None,
        })
    return nodes

def find_node_by_det(diagram_nodes, det, tol=8):
    if det is None:
        return None
    label, tl, br = det[0], det[1], det[2]
    label_snake = _to_snake_case(label)
    cx = int(round((tl[0] + br[0]) / 2))
    cy = int(round((tl[1] + br[1]) / 2))
    for node in diagram_nodes:
        if node['shape'] == label_snake and abs(node['x'] - cx) <= tol and abs(node['y'] - cy) <= tol:
            return node['id']
    best = (None, 1e9)
    for node in diagram_nodes:
        if node['shape'] == label_snake:
            d = (node['x'] - cx) ** 2 + (node['y'] - cy) ** 2
            if d < best[1]:
                best = (node['id'], d)
    return best[0]

def attach_angles_to_nodes(diagram_nodes, detections, angle_map):
    if not diagram_nodes or not angle_map:
        return diagram_nodes
    id_to_idx = {n['id']: idx for idx, n in enumerate(diagram_nodes)}
    for det, angle in angle_map.items():
        if angle is None:
            continue
        node_id = find_node_by_det(diagram_nodes, det)
        if not node_id:
            continue
        idx = id_to_idx.get(node_id)
        if idx is None:
            continue
        # Normalize to an integer degree if possible
        val = None
        if isinstance(angle, (int, float)):
            val = int(round(angle))
        elif isinstance(angle, str):
            import re
            m = re.search(r'-?\d+', angle)
            if m:
                val = int(m.group(0))
        if val is None:
            continue  # keep the existing default (0)
        diagram_nodes[idx]['angle'] = val
    return diagram_nodes
