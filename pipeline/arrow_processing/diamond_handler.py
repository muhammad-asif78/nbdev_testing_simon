import numpy as np
import cv2

def dilate_ocr_regions(binary_mask, ocr_intrusions, crop_origin, vert_expand=1, horiz_expand=1, vert_iterations=1, horiz_iterations=1, erosion_kernel_size=3, erosion_iterations=1):
    mask = np.zeros_like(binary_mask)
    crop_x1, crop_y1 = crop_origin
    h, w = mask.shape
    for ocr in ocr_intrusions:
        ox1, oy1, ox2, oy2 = ocr['bbox']
        local_x1 = max(0, int(ox1 - crop_x1))
        local_y1 = max(0, int(oy1 - crop_y1))
        local_x2 = min(w, int(ox2 - crop_x1))
        local_y2 = min(h, int(oy2 - crop_y1))
        if local_x1 < local_x2 and local_y1 < local_y2:
            mask[local_y1:local_y2, local_x1:local_x2] = 255

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_expand))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_expand, 1))

    mask = cv2.dilate(mask, vert_kernel, iterations=vert_iterations)
    mask = cv2.dilate(mask, horiz_kernel, iterations=horiz_iterations)

    combined = cv2.bitwise_or(binary_mask, mask)

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
    combined = cv2.erode(combined, erosion_kernel, iterations=erosion_iterations)

    return combined

def find_skeleton_points_on_bbox(skeleton, bbox_tl, bbox_br, margin=2):
    points = np.column_stack(np.where(skeleton > 0))
    x_min, y_min = bbox_tl
    x_max, y_max = bbox_br
    candidate_points = []
    for y, x in points:
        near_left = abs(x - x_min) <= margin and y_min <= y <= y_max
        near_right = abs(x - x_max) <= margin and y_min <= y <= y_max
        near_top = abs(y - y_min) <= margin and x_min <= x <= x_max
        near_bottom = abs(y - y_max) <= margin and x_min <= x <= x_max
        if near_left or near_right or near_top or near_bottom:
            candidate_points.append((x, y))
    return candidate_points
