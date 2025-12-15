import numpy as np
from .node_builder import find_node_by_det

def create_edges_json(arrow_connections, diagram_nodes, all_dets_with_color, arrowhead_radius=50):
    edges = []
    arrow_head_centers = []
    for d in all_dets_with_color:
        if d[0] == "arrow_head":
            lab, tl, br = d[0], d[1], d[2]
            arrow_head_centers.append(((tl[0] + br[0]) / 2.0, (tl[1] + br[1]) / 2.0))

    def _has_arrow_at(pt):
        if pt is None:
            return False
        for ah in arrow_head_centers:
            if np.linalg.norm(np.array(pt) - np.array(ah)) < arrowhead_radius:
                return True
        return False

    edge_counter = 1
    for i, conn in enumerate(arrow_connections):
        tail_det = conn.get("tail_connected_to")
        head_det = conn.get("head_connected_to")
        src_id = find_node_by_det(diagram_nodes, tail_det)
        tgt_id = find_node_by_det(diagram_nodes, head_det)
        if not src_id or not tgt_id:
            continue
        head_pt = tuple(conn.get('head')) if conn.get('head') is not None else None
        tail_pt = tuple(conn.get('tail')) if conn.get('tail') is not None else None
        has_tail = _has_arrow_at(tail_pt)
        has_head = _has_arrow_at(head_pt)
        startArrow = False
        endArrow = False
        if has_tail and has_head:
            startArrow = True
            endArrow = True
        elif has_tail and not has_head:
            src_id, tgt_id = tgt_id, src_id
            endArrow = True
        elif has_head and not has_tail:
            endArrow = True
        else:
            if head_pt and tail_pt:
                x_diff = head_pt[0] - tail_pt[0]
                y_diff = head_pt[1] - tail_pt[1]
                if abs(x_diff) > abs(y_diff):
                    if x_diff < 0:
                        src_id, tgt_id = tgt_id, src_id
                else:
                    if y_diff < 0:
                        src_id, tgt_id = tgt_id, src_id
        raw = conn.get("original_label") or conn.get("label") or ""
        line_style = raw.split('-')[0] if raw else "solid"
        edges.append({
            "id": f"edge{edge_counter}",
            "source": src_id,
            "target": tgt_id,
            "lineStyle": line_style,
            "startArrow": bool(startArrow),
            "endArrow": bool(endArrow),
            "color": "#333333",
        })
        edge_counter += 1
    return edges
