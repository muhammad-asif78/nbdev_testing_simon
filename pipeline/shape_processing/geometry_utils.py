import math
import cv2
import numpy as np

def check_bbox_intersection(box1_tl, box1_br, box2_tl, box2_br):
    return not (box1_br[0] < box2_tl[0] or box1_tl[0] > box2_br[0] or box1_br[1] < box2_tl[1] or box1_tl[1] > box2_br[1])

def nearest_point_on_bbox(point, top_left, bottom_right):
    px, py = point
    x1, y1 = top_left
    x2, y2 = bottom_right
    cx = min(max(px, x1), x2)
    cy = min(max(py, y1), y2)
    return (int(round(cx)), int(round(cy)))

def find_nearest_shape_bbox(point, shapes, max_distance=None):
    if not shapes:
        return None, None, None
    best_d = float('inf')
    best_shape = None
    best_pt = None
    for shape in shapes:
        label, tl, br = shape[0], shape[1], shape[2]
        pt_on = nearest_point_on_bbox(point, tl, br)
        d = math.hypot(pt_on[0] - point[0], pt_on[1] - point[1])
        if d < best_d:
            best_d, best_shape, best_pt = d, shape, pt_on
    if max_distance is not None and best_d > max_distance:
        return None, None, None
    return best_shape, best_pt, best_d

def find_farthest_points_in_contour(contour):
    max_dist_sq, best_pair = -1, None
    if contour.ndim < 3 or contour.shape[1] == 0:
        return None
    points = contour.squeeze(axis=1)
    if points.ndim != 2 or len(points) < 2:
        return None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, p2 = points[i], points[j]
            d2 = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
            if d2 > max_dist_sq:
                max_dist_sq, best_pair = d2, (tuple(p1), tuple(p2))
    return best_pair

def find_farthest_points_euclidean(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    valid = []
    for cnt in contours:
        if cnt.ndim < 3 or cnt.shape[1] == 0:
            continue
        squeezed = cnt.squeeze(axis=1)
        if squeezed.ndim != 2 or len(squeezed) == 0:
            continue
        valid.append(squeezed)
    if not valid:
        return []
    all_points = np.vstack(valid)
    if len(all_points) < 2:
        return []
    return find_farthest_points_in_contour(np.array(all_points).reshape(-1, 1, 2))

def establish_connections(vectors, detections):
    labels_to_exclude = {"dashed-arrow", "dotted-arrow", "solid-arrow", "arrow_head"}
    shapes = [d for d in detections if d[0] not in labels_to_exclude]
    conns = []
    for vec in vectors:
        head_pt = vec["head"]
        tail_pt = vec["tail"]
        MAXD = 44
        head_shape, head_on, head_d = find_nearest_shape_bbox(head_pt, shapes, max_distance=MAXD)
        tail_shape, tail_on, tail_d = find_nearest_shape_bbox(tail_pt, shapes, max_distance=MAXD)
        conns.append({
            "head": head_pt,
            "tail": tail_pt,
            "head_connected_to": head_shape,
            "head_connection_point": head_on,
            "head_connection_dist": head_d,
            "tail_connected_to": tail_shape,
            "tail_connection_point": tail_on,
            "tail_connection_dist": tail_d,
            "original_label": vec.get("label", ""),
        })
    return conns
