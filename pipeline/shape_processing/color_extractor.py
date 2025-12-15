import cv2
import numpy as np


def find_dominant_color_masked(image_crop, mask=None, k=3):
    if image_crop is None or image_crop.size == 0:
        return None

    try:
        lab_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2LAB)
        
        h, w = lab_image.shape[:2]
        corners = np.array([
            lab_image[0, 0],       
            lab_image[0, w-1],     
            lab_image[h-1, 0],     
            lab_image[h-1, w-1]    
        ])
        
        bg_lab_mean = np.median(corners, axis=0)

        if mask is not None:
            valid_pixels = mask > 0
            if not np.any(valid_pixels): 
                return None
            pixels = lab_image[valid_pixels].reshape((-1, 3)).astype(np.float32)
        else:
            pixels = lab_image.reshape((-1, 3)).astype(np.float32)

        if pixels.shape[0] < 10: 
            return None

        K = min(k, max(1, pixels.shape[0] // 50))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        counts = np.bincount(labels.flatten())
        sorted_indices = np.argsort(counts)[::-1]

        final_color_lab = None
        
        for idx in sorted_indices:
            candidate_lab = centers[idx]
            dist_to_bg = np.linalg.norm(candidate_lab - bg_lab_mean)
            
            if dist_to_bg > 10.0:
                final_color_lab = candidate_lab
                break
        
        if final_color_lab is None:
            final_color_lab = centers[sorted_indices[0]]

        final_color_lab_u8 = np.uint8([[final_color_lab]])
        final_color_bgr = cv2.cvtColor(final_color_lab_u8, cv2.COLOR_LAB2BGR)[0][0]
        
        return tuple(int(c) for c in final_color_bgr[::-1]) 

    except Exception:
        return None


def rgb_to_hex(rgb_tuple):
    if rgb_tuple is None:
        return None
    r, g, b = rgb_tuple
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, int(r))),
        max(0, min(255, int(g))),
        max(0, min(255, int(b)))
    )


def get_corner_background_color(image):
    h, w = image.shape[:2]
    s = min(5, w // 2, h // 2)
    if s < 1:
        return np.array([255, 255, 255], dtype=float)

    tl = image[0:s, 0:s].reshape(-1, 3)
    tr = image[0:s, w-s:w].reshape(-1, 3)
    bl = image[h-s:h, 0:s].reshape(-1, 3)
    br = image[h-s:h, w-s:w].reshape(-1, 3)
    corners = np.vstack([tl, tr, bl, br])
    mean_bgr = np.mean(corners, axis=0)
    return mean_bgr


def extract_shape_mask(H, W, shape_type, bbox, vector_data=None):
    mask = np.zeros((H, W), dtype=np.uint8)
    x, y, w, h = bbox

    if shape_type == "rectangle":
        mask[y:y+h, x:x+w] = 1

    elif shape_type == "rounded rectangle":
        radius = min(w, h) // 8
        cv2.rectangle(mask, (x, y), (x+w, y+h), 1, -1)
        cv2.circle(mask, (x+radius, y+radius), radius, 1, -1)
        cv2.circle(mask, (x+w-radius, y+radius), radius, 1, -1)
        cv2.circle(mask, (x+radius, y+h-radius), radius, 1, -1)
        cv2.circle(mask, (x+w-radius, y+h-radius), radius, 1, -1)

    elif shape_type == "circle":
        center = (x+w//2, y+h//2)
        radius = min(w, h)//2
        cv2.circle(mask, center, radius, 1, -1)

    elif shape_type == "diamond":
        pts = np.array([
            [x+w//2, y],
            [x+w-1, y+h//2],
            [x+w//2, y+h-1],
            [x, y+h//2]
        ], np.int32)
        cv2.fillConvexPoly(mask, pts, 1)

    elif shape_type == "triangle":
        pts = np.array([
            [x+w//2, y],
            [x+w-1, y+h-1],
            [x, y+h-1]
        ], np.int32)
        cv2.fillConvexPoly(mask, pts, 1)

    elif shape_type == "pentagon":
        pts = np.array([
            [x+w//2, y],
            [x+w-1, y+h//3],
            [x+3*w//4, y+h-1],
            [x+w//4, y+h-1],
            [x, y+h//3]
        ], np.int32)
        cv2.fillConvexPoly(mask, pts, 1)

    elif shape_type == "star":
        if vector_data is not None:
            cv2.fillPoly(mask, [np.array(vector_data, np.int32)], 1)

    elif shape_type == "cloud":
        if vector_data is not None:
            cv2.fillPoly(mask, [np.array(vector_data, np.int32)], 1)

    elif shape_type in ["arrow", "double arrow"]:
        if vector_data is not None:
            cv2.fillPoly(mask, [np.array(vector_data, np.int32)], 1)

    elif shape_type == "racetrack":
        if vector_data is not None:
            cv2.fillPoly(mask, [np.array(vector_data, np.int32)], 1)
        else:
            mask[y:y+h, x:x+w] = 1

    elif shape_type == "sticky notes":
        mask[y:y+h, x:x+w] = 1

    else:
        mask[y:y+h, x:x+w] = 1

    return mask


def find_dominant_color(image, shape_type="rectangle", bbox=None, k=4, least_prominent=False, bg_rgb=(255,255,255), bg_thresh=24):
    try:
        H, W = image.shape[:2]
        if bbox is not None:
            mask = extract_shape_mask(H, W, shape_type, bbox)
            crop = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            mask_crop = mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        else:
            crop = image
            mask_crop = np.ones(crop.shape[:2], dtype=np.uint8)

        pixels = crop[mask_crop > 0].astype(np.float32)
        if len(pixels) < 10:
            mean_bgr = crop.reshape(-1, 3).mean(axis=0).astype(int)
            return (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))
        
        dist = np.linalg.norm(pixels - np.array(bg_rgb), axis=1)
        pixels = pixels[dist > bg_thresh]
        if len(pixels) < 10: 
            mean_bgr = crop.reshape(-1, 3).mean(axis=0).astype(int)
            return (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))

        lab_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)
        K = min(k, max(1, len(lab_pixels) // 200))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            lab_pixels, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        counts = np.bincount(labels.flatten())
        chosen_idx = int(np.argmin(counts)) if least_prominent else int(np.argmax(counts))
        dominant_color_lab = centers[chosen_idx].astype("uint8").reshape(1, 1, 3)
        dominant_color_bgr = cv2.cvtColor(dominant_color_lab, cv2.COLOR_LAB2BGR).reshape(3)
        return tuple(int(c) for c in dominant_color_bgr[::-1])  
    except Exception:
        return None


def attach_colors(image_bgr, detections):

    H, W = image_bgr.shape[:2]
    enriched = []
    
    std_dets = []
    for i, det in enumerate(detections):
        if len(det) >= 4:
            label, tl, br, confidence = det[0], det[1], det[2], det[3]
        else:
            label, tl, br = det[0], det[1], det[2]
            confidence = None
            
        x1 = max(0, int(round(float(tl[0]))))
        y1 = max(0, int(round(float(tl[1]))))
        x2 = min(W, int(round(float(br[0]))))
        y2 = min(H, int(round(float(br[1]))))
        
        std_dets.append({
            'i': i,
            'label': label,
            'tl': tl,
            'br': br,
            'box': (x1, y1, x2, y2),
            'confidence': confidence
        })
    
    for curr in std_dets:
        cx1, cy1, cx2, cy2 = curr['box']
        
        hex_color = None
        
        if cx2 > cx1 and cy2 > cy1:
            full_crop = image_bgr[cy1:cy2, cx1:cx2]
            
            if full_crop is not None and full_crop.size > 0:
                crop_h, crop_w = full_crop.shape[:2]
                
                if curr['label'].lower() == 'triangle':
                    center_y1 = 2 * crop_h // 5
                    center_y2 = 3 * crop_h // 5
                    center_x1 = 2 * crop_w // 5
                    center_x2 = 3 * crop_w // 5
                    center_region = full_crop[center_y1:center_y2, center_x1:center_x2]
                    if center_region.size > 0:
                        pixels = center_region.reshape(-1, 3)
                        brightness = pixels.sum(axis=1)
                        brightest_idx = np.argmax(brightness)
                        brightest_bgr = pixels[brightest_idx]
                        hex_color = rgb_to_hex(tuple(int(c) for c in brightest_bgr[::-1]))
                else:
                    mask = np.ones((crop_h, crop_w), dtype=np.uint8) * 255
                    
                    for other in std_dets:
                        if curr['i'] == other['i']:
                            continue   
                        
                        ox1, oy1, ox2, oy2 = other['box']
                        
                        curr_area = (cx2 - cx1) * (cy2 - cy1)
                        other_area = (ox2 - ox1) * (oy2 - oy1)
                        if other_area >= curr_area:
                            continue
                        
                        ix1 = max(cx1, ox1)
                        iy1 = max(cy1, oy1)
                        ix2 = min(cx2, ox2)
                        iy2 = min(cy2, oy2)
                        
                        if ix2 > ix1 and iy2 > iy1:
                            rel_x1 = max(0, ix1 - cx1)
                            rel_y1 = max(0, iy1 - cy1)
                            rel_x2 = min(crop_w, ix2 - cx1)
                            rel_y2 = min(crop_h, iy2 - cy1)
                            
                            cv2.rectangle(mask, (rel_x1, rel_y1), (rel_x2, rel_y2), 0, -1)
                    
                    dom_rgb = find_dominant_color_masked(full_crop, mask=mask, k=3)
                    if dom_rgb:
                        hex_color = rgb_to_hex(dom_rgb)
        
        if hex_color is None:
            hex_color = "#cccccc"
        
        enriched.append((curr['label'], curr['tl'], curr['br'], hex_color, curr['confidence']))
    
    return enriched
