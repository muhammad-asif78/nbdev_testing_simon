import math
import cv2
import numpy as np
from .skeleton_analyzer import extract_skeleton, skeleton_has_cycle, find_best_arrow_by_straightness
from .diamond_handler import dilate_ocr_regions, find_skeleton_points_on_bbox
from ..shape_processing.geometry_utils import check_bbox_intersection, find_farthest_points_euclidean

def crop_image_region(image, top_left, bottom_right):
    x1, y1 = map(int, map(round, top_left))
    x2, y2 = map(int, map(round, bottom_right))
    return image[y1:y2, x1:x2]

def convert_to_binary_mask(cropped_img):
    if cropped_img.size == 0:
        return np.array([], dtype=np.uint8)
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    return cv2.bitwise_or(closed, thresh)

def find_arrow_endpoints(image_bgr, detections, ocr_results):
    vectors = []
    arrow_labels = {"dashed-arrow", "dotted-arrow", "solid-arrow"}
    arrows = [d for d in detections if d[0] in arrow_labels]

    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        xA = max(ax1, bx1)
        yA = max(ay1, by1)
        xB = min(ax2, bx2)
        yB = min(ay2, by2)
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        aA = (ax2 - ax1) * (ay2 - ay1)
        bA = (bx2 - bx1) * (by2 - by1)
        return inter / float(aA + bA - inter)

    def _pixel_overlap(mask, ocr_bbox, origin, th=0.02):
        ox1, oy1, ox2, oy2 = ocr_bbox
        cx, cy = origin
        h, w = mask.shape
        lx1 = max(0, int(ox1 - cx))
        ly1 = max(0, int(oy1 - cy))
        lx2 = min(w, int(ox2 - cx))
        ly2 = min(h, int(oy2 - cy))
        if lx1 >= lx2 or ly1 >= ly2:
            return False
        arrow_pixels = np.count_nonzero(mask)
        if arrow_pixels == 0:
            return False
        overlap = mask[ly1:ly2, lx1:lx2]
        return (np.count_nonzero(overlap) / arrow_pixels) > th

    def _mask_ocr(mask, intrusions, origin):
        out = mask.copy()
        cx, cy = origin
        h, w = out.shape
        for o in intrusions:
            ox1, oy1, ox2, oy2 = o['bbox']
            lx1 = max(0, int(ox1 - cx))
            ly1 = max(0, int(oy1 - cy))
            lx2 = min(w, int(ox2 - cx))
            ly2 = min(h, int(oy2 - cy))
            if lx1 < lx2 and ly1 < ly2:
                out[ly1:ly2, lx1:lx2] = 0
        return out

    def _bridge(mask, local_bbox, thickness=1):
        h, w = mask.shape[:2]
        x1, y1, x2, y2 = map(int, local_bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return mask.copy()
        mask_no_text = mask.copy()
        mask_no_text[y1:y2, x1:x2] = 0
        pts = np.column_stack(np.where(mask_no_text > 0))
        if pts.size == 0:
            return mask.copy()
        left = pts[pts[:, 1] < x1]
        right = pts[pts[:, 1] > x2]
        if left.size and right.size:
            lp = (int(left[np.argmax(left[:, 1]), 1]), int(left[np.argmax(left[:, 1]), 0]))
            rp = (int(right[np.argmin(right[:, 1]), 1]), int(right[np.argmin(right[:, 1]), 0]))
            p1, p2 = lp, rp
        else:
            top = pts[pts[:, 0] < y1]
            bottom = pts[pts[:, 0] > y2]
            if top.size and bottom.size:
                tp = (int(top[np.argmax(top[:, 0]), 1]), int(top[np.argmax(top[:, 0]), 0]))
                bp = (int(bottom[np.argmin(bottom[:, 0]), 1]), int(bottom[np.argmin(bottom[:, 0]), 0]))
                p1, p2 = tp, bp
            else:
                coords = np.column_stack((pts[:, 1], pts[:, 0]))
                max_d, best = -1, None
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        d = (coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2
                        if d > max_d:
                            max_d, best = d, (tuple(coords[i].tolist()), tuple(coords[j].tolist()))
                if not best:
                    return mask.copy()
                p1, p2 = best
        out = mask_no_text.copy()
        dist = int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if dist < 2:
            cv2.line(out, p1, p2, 255, thickness=max(1, thickness))
            return out
        n = max(dist, 2)
        xs = np.linspace(p1[0], p2[0], n)
        ys = np.linspace(p1[1], p2[1], n)
        for x, y in zip(xs, ys):
            cx, cy = int(round(x)), int(round(y))
            x0, x1_ = max(0, cx - thickness), min(w, cx + thickness + 1)
            y0, y1_ = max(0, cy - thickness), min(h, cy + thickness + 1)
            out[y0:y1_, x0:x1_] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.dilate(out, kernel, iterations=1)

    def _interpolate_dotted(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.dilate(closed, kernel, iterations=2)

    def _interpolate_dashed(mask):
        from ..shape_processing.geometry_utils import find_farthest_points_in_contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 3]
        if len(contours) < 2:
            return mask
        dash_endpoints = []
        for cnt in contours:
            ep = find_farthest_points_in_contour(cnt)
            if ep:
                dash_endpoints.append(list(ep))
        lengths = [math.dist(e[0], e[1]) for e in dash_endpoints if e]
        if not lengths:
            return mask
        max_connect = np.median(lengths) * 1.5
        out = mask.copy()
        for i, eps1 in enumerate(dash_endpoints):
            for ep1 in eps1:
                min_d = float('inf')
                best = None
                for j, eps2 in enumerate(dash_endpoints):
                    if i == j:
                        continue
                    for ep2 in eps2:
                        d = math.dist(ep1, ep2)
                        if d < min_d:
                            min_d, best = d, ep2
                if best and min_d < max_connect:
                    cv2.line(out, ep1, best, 255, thickness=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.dilate(out, kernel, iterations=2)

    for arrow in arrows:
        label, tl, br = arrow[0], arrow[1], arrow[2]
        crop = crop_image_region(image_bgr, tl, br)
        if crop.size == 0:
            continue
        binary = convert_to_binary_mask(crop)
        if binary.size == 0:
            continue
        processed = binary.copy()
        is_overlap = any(check_bbox_intersection(tl, br, a[1], a[2]) for a in arrows if a != arrow)
        intrusions = []
        if ocr_results:
            for o in ocr_results:
                iou = _bbox_iou((tl[0], tl[1], br[0], br[1]), o['bbox'])
                if iou > 0.02 and _pixel_overlap(processed, o['bbox'], (int(tl[0]), int(tl[1])), th=0.02):
                    intrusions.append(o)
        ocr_interrupted = len(intrusions) > 0
        
        if not ocr_interrupted:
            if label == "dashed-arrow":
                processed = _interpolate_dashed(processed)
            elif label == "dotted-arrow":
                processed = _interpolate_dotted(processed)
        
        if ocr_interrupted:
            processed = _mask_ocr(processed, intrusions, (int(tl[0]), int(tl[1])))
            
            processed = dilate_ocr_regions(
                processed, intrusions, (int(tl[0]), int(tl[1])),
                vert_expand=5, horiz_expand=10, vert_iterations=3, horiz_iterations=1
            )
            processed = _interpolate_dotted(processed)  
            
            skeleton = extract_skeleton(processed)
            if skeleton_has_cycle(skeleton):
                diamond_labels = ["Diamond"]
                interfering_objects = [d for d in detections if d[0] in diamond_labels]
                intersecting_shapes = [d for d in interfering_objects if check_bbox_intersection(tl, br, d[1], d[2])]
                
                all_intersection_points = []
                crop_x1, crop_y1 = int(tl[0]), int(tl[1])
                for shape in intersecting_shapes:
                    shape_label, shape_tl, shape_br = shape[0], shape[1], shape[2]
                    local_shape_tl = (shape_tl[0] - crop_x1, shape_tl[1] - crop_y1)
                    local_shape_br = (shape_br[0] - crop_x1, shape_br[1] - crop_y1)
                    points_on_bbox = find_skeleton_points_on_bbox(
                        skeleton, local_shape_tl, local_shape_br, margin=2
                    )
                    all_intersection_points.extend(points_on_bbox)
                
                if len(all_intersection_points) >= 2:
                    max_dist = -1
                    endpoints = None
                    import itertools
                    for p1, p2 in itertools.combinations(all_intersection_points, 2):
                        dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                        if dist > max_dist:
                            max_dist = dist
                            endpoints = [p1, p2]
                else:
                    endpoints = find_best_arrow_by_straightness(processed)
            else:
                endpoints = find_best_arrow_by_straightness(processed)
                if not endpoints or len(endpoints) < 2:
                    endpoints = find_farthest_points_euclidean(processed)
        elif is_overlap:
            endpoints = find_best_arrow_by_straightness(processed)
            if not endpoints:
                endpoints = find_farthest_points_euclidean(processed)
        else:
            endpoints = find_best_arrow_by_straightness(processed)
        
        if not endpoints:
            endpoints = find_farthest_points_euclidean(processed)
        if not endpoints or len(endpoints) < 2:
            continue
        w = br[0] - tl[0]
        h = br[1] - tl[1]
        if w >= h:
            tail_local, head_local = sorted(endpoints, key=lambda e: e[0])
        else:
            tail_local, head_local = sorted(endpoints, key=lambda e: e[1])
        cx, cy = int(tl[0]), int(tl[1])
        vectors.append({
            "tail": (tail_local[0] + cx, tail_local[1] + cy),
            "head": (head_local[0] + cx, head_local[1] + cy),
            "bbox": (tl, br),
            "label": label,
        })
    return vectors
