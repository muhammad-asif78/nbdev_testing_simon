import os
from typing import List, Dict, Any
import numpy as np
import cv2
from .angle_predictor import _load_angle_models, _get_angle_transform, _TORCH_AVAILABLE

_SAM2_AVAILABLE = True
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:
    _SAM2_AVAILABLE = False
_SAM2_PREDICTOR = None  

def _boxes_overlap(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if ax2 < bx1 or bx2 < ax1:
        return False
    if ay2 < by1 or by2 < ay1:
        return False
    return True

def _load_sam2_predictor():
    global _SAM2_PREDICTOR
    if _SAM2_PREDICTOR is not None:
        return _SAM2_PREDICTOR
    if not _SAM2_AVAILABLE:
        return None
    try:
        cfg_env = os.getenv("SAM2_CONFIG")
        cfg = None
        if cfg_env and os.path.exists(cfg_env):
            cfg = cfg_env
        else:
            local_cfg = os.path.join("sam2", "configs", "sam2", "sam2_hiera_b+.yaml")
            if os.path.exists(local_cfg):
                cfg = local_cfg
            else:
                try:
                    import sam2 as _sam2_pkg  
                    from pathlib import Path
                    base = Path(os.path.dirname(_sam2_pkg.__file__))
                    pkg_cfg = base / "configs" / "sam2" / "sam2_hiera_b+.yaml"
                    if pkg_cfg.exists():
                        cfg = str(pkg_cfg)
                except Exception:
                    cfg = None
        if not cfg:
            cfg = os.path.join("sam2", "configs", "sam2", "sam2_hiera_b+.yaml")

        ckpt = os.getenv("SAM2_CHECKPOINT", os.path.join("sam2_checkpoints", "sam2_hiera_base_plus.pt"))
        device = os.getenv("SAM2_DEVICE", "cpu")
        model = build_sam2(cfg, ckpt, device=device)
        predictor = SAM2ImagePredictor(model)
        _SAM2_PREDICTOR = (predictor, device)
        return _SAM2_PREDICTOR
    except Exception:
        _SAM2_PREDICTOR = None
        return None

def _compute_occlusion_flags(filtered_dets, boxes):
    completable_shapes = {"circle", "rounded rectangle", "rectangle", "Racetrack"}
    flags = []
    for i in range(len(filtered_dets)):
        is_occ = False
        label_i = filtered_dets[i][0]
        box_i = boxes[i]
        if label_i in completable_shapes:
            for j in range(len(filtered_dets)):
                if i == j:
                    continue
                if _boxes_overlap(box_i, boxes[j]):
                    is_occ = True
                    break
        flags.append(is_occ)
    return flags

def _postprocess_mask(label: str, mask_binary, is_occluded: bool):
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_binary
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask_binary)
    if is_occluded:
        if label == 'circle':
            (x, y), radius = cv2.minEnclosingCircle(largest)
            cv2.circle(out, (int(x), int(y)), int(radius), 1, thickness=cv2.FILLED)
        elif label == 'Racetrack':
            rect = cv2.minAreaRect(largest)
            center, (width, height), angle = rect
            if width < height:
                width, height = height, width
                angle += 90
            radius = height / 2
            if width <= height:
                cv2.circle(out, (int(center[0]), int(center[1])), int(radius), 1, thickness=cv2.FILLED)
            else:
                rect_center = np.array(center)
                angle_rad = np.deg2rad(angle)
                half_rect_len = width / 2 - radius
                dx = np.cos(angle_rad) * half_rect_len
                dy = np.sin(angle_rad) * half_rect_len
                p1 = rect_center + np.array([dx, dy])
                p2 = rect_center - np.array([dx, dy])
                cv2.circle(out, (int(p1[0]), int(p1[1])), int(radius), 1, thickness=cv2.FILLED)
                cv2.circle(out, (int(p2[0]), int(p2[1])), int(radius), 1, thickness=cv2.FILLED)
                perp_dx = -np.sin(angle_rad) * radius
                perp_dy = np.cos(angle_rad) * radius
                corner1 = p1 + np.array([perp_dx, perp_dy])
                corner2 = p1 - np.array([perp_dx, perp_dy])
                corner3 = p2 - np.array([perp_dx, perp_dy])
                corner4 = p2 + np.array([perp_dx, perp_dy])
                rect_contour = np.array([corner1, corner2, corner3, corner4], dtype=np.intp)
                cv2.drawContours(out, [rect_contour], 0, 1, thickness=cv2.FILLED)
        else:
            hull = cv2.convexHull(largest)
            cv2.drawContours(out, [hull], -1, 1, thickness=cv2.FILLED)
    else:
        cv2.drawContours(out, [largest], -1, 1, thickness=cv2.FILLED)
    return out

def _mask_to_padded_rgb(mask_binary, pad_pct: float = 0.1):
    from PIL import Image
    if mask_binary is None or mask_binary.ndim < 2:
        return None
    rows = np.any(mask_binary, axis=1) if mask_binary.shape[0] > 0 else np.array([], dtype=bool)
    cols = np.any(mask_binary, axis=0) if mask_binary.shape[1] > 0 else np.array([], dtype=bool)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    height_pad = rmax - rmin
    width_pad = cmax - cmin
    py = int(height_pad * pad_pct)
    px = int(width_pad * pad_pct)
    h, w = mask_binary.shape
    r0 = max(0, rmin - py)
    r1 = min(h - 1, rmax + py)
    c0 = max(0, cmin - px)
    c1 = min(w - 1, cmax + px)
    crop = mask_binary[r0:r1 + 1, c0:c1 + 1]
    if crop.size == 0:
        return None
    final_rgb = np.zeros((crop.shape[0], crop.shape[1], 3), dtype=np.uint8)
    color = np.random.randint(0, 256, size=3)
    for ch in range(3):
        final_rgb[:, :, ch][crop == 1] = int(color[ch])
    return Image.fromarray(final_rgb)

def predict_angles_with_masks(image_bgr, detections):
    from PIL import Image
    import torch
    
    include_labels = {"Pentagon", "Racetrack", "Triangle", "arrow", "rectangle"}
    filtered = [d for d in detections if d[0] in include_labels]
    if not filtered:
        return {}
    sam2_loaded = _load_sam2_predictor()
    if not sam2_loaded:
        return {}
    predictor, device = sam2_loaded
    boxes = []
    for det in filtered:
        label, tl, br = det[0], det[1], det[2]
        x1, y1 = tl
        x2, y2 = br
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    try:
        if _TORCH_AVAILABLE:
            ctx = torch.inference_mode()
            ctx.__enter__()
            if device == "cuda":
                ac = torch.autocast("cuda", dtype=torch.bfloat16)
                ac.__enter__()
        predictor.set_image(pil_img)
        masks, iou_preds, _ = predictor.predict(box=boxes)
    finally:
        if _TORCH_AVAILABLE:
            try:
                if device == "cuda":
                    ac.__exit__(None, None, None)
            except Exception:
                pass
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
    models_dict = _load_angle_models()
    
    triangle_aux = None
    if _TORCH_AVAILABLE:
        from torchvision import models
        aux_path = os.getenv("ANGLE_TRIANGLE_AUX", "weights/angle-models/Triangle_Angle_90_270.pth")
        if os.path.exists(aux_path):
            try:
                m = models.resnet18(pretrained=False)
                m.fc = torch.nn.Linear(m.fc.in_features, 2)
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                state = torch.load(aux_path, map_location=dev)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                m.load_state_dict(state, strict=False)
                m.to(dev)
                m.eval()
                triangle_aux = (m, dev)
            except Exception:
                triangle_aux = None
    out = {}
    transform = _get_angle_transform()
    occ_flags = _compute_occlusion_flags(filtered, boxes)
    for i, det in enumerate(filtered):
        label, tl, br = det[0], det[1], det[2]
        if masks is None or i >= len(masks):
            continue
        box_masks = masks[i]
        scores = iou_preds[i] if i < len(iou_preds) else None
        if box_masks is None:
            continue
        if scores is not None:
            try:
                if _TORCH_AVAILABLE:
                    best_idx = int(torch.argmax(torch.from_numpy(np.asarray(scores))).item())
                else:
                    best_idx = int(np.argmax(scores))
            except Exception:
                best_idx = 0
        else:
            best_idx = 0
        best_mask = box_masks[best_idx]
        mask_binary = (best_mask > 0).astype(np.uint8)
        my_box = boxes[i]
        occ = bool(occ_flags[i])
        processed = _postprocess_mask(label, mask_binary, occ)
        pil_crop = _mask_to_padded_rgb(processed, pad_pct=0.1)
        if not pil_crop or not models_dict or label not in models_dict or transform is None:
            out[det] = ""
            continue
        model, mdev = models_dict[label]
        tens = transform(pil_crop).unsqueeze(0)
        cls_idx = None
        try:
            with torch.no_grad():
                logits = model(tens.to(mdev))
                cls_idx = int(torch.argmax(logits, dim=1).item())
        except Exception:
            out[det] = ""
            continue
        
        angle_val = None
        if label == 'Pentagon':
            angle_val = 0 if cls_idx == 0 else (180 if cls_idx == 1 else None)
        elif label == 'rectangle':
            x1, y1 = my_box[0], my_box[1]
            x2, y2 = my_box[2], my_box[3]
            w = x2 - x1
            h = y2 - y1
            if abs(w - h) < 5:
                angle_val = 0
            elif h >= 2 * w:
                angle_val = 90
            else:
                angle_val = 0
        elif label == 'Racetrack':
            mapping = {0: 45, 1: 90, 2: 135, 3: 0}
            angle_val = mapping.get(cls_idx)
        elif label == 'Triangle':
            if cls_idx == 0:
                angle_val = 0
            elif cls_idx == 2:
                angle_val = 180
            else:
                if triangle_aux is not None and transform is not None:
                    aux_m, aux_dev = triangle_aux
                    try:
                        with torch.no_grad():
                            aux_logits = aux_m(transform(pil_crop).unsqueeze(0).to(aux_dev))
                            aux_idx = int(torch.argmax(aux_logits, dim=1).item())
                        if aux_idx == 0:
                            angle_val = 90
                        elif aux_idx == 1:
                            angle_val = 270
                        else:
                            angle_val = None
                    except Exception:
                        angle_val = None
                else:
                    angle_val = None
        elif label == 'arrow':
            mapping = {0: 0, 1: 90, 2: 180, 3: 270}
            angle_val = mapping.get(cls_idx)
        else:
            from .angle_predictor import _ANGLE_NUM_CLASSES
            nc = _ANGLE_NUM_CLASSES.get(label)
            if nc == 8:
                mp = {0: 0, 1: 45, 2: 90, 3: 135, 4: 180, 5: 225, 6: 270, 7: 315}
                angle_val = mp.get(cls_idx)
            elif nc == 4:
                mp = {0: 0, 1: 90, 2: 180, 3: 270}
                angle_val = mp.get(cls_idx)
            elif nc == 2:
                mp = {0: 0, 1: 180}
                angle_val = mp.get(cls_idx)
            else:
                angle_val = None
        out[det] = angle_val
    return out
