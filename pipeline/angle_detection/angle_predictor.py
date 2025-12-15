# import os
# from typing import List, Dict, Any
# import numpy as np

# _TORCH_AVAILABLE = True
# try:
#     import torch
#     from torchvision import models, transforms
#     from PIL import Image
# except Exception:
#     _TORCH_AVAILABLE = False

# _ANGLE_ALLOWED = {"rectangle", "Triangle", "Racetrack", "Pentagon", "arrow"}
# _ANGLE_MODEL_PATHS = {
#     "rectangle": os.getenv("ANGLE_RECTANGLE", "weights/angle-models/final_rectangle.pth"),
#     "Triangle": os.getenv("ANGLE_TRIANGLE", "weights/angle-models/Triangle.pth"),
#     "Racetrack": os.getenv("ANGLE_RACETRACK", "weights/angle-models/final_racetrack.pth"),
#     "Pentagon": os.getenv("ANGLE_PENTAGON", "weights/angle-models/best_resnet18_pentagon.pth"),
#     "arrow": os.getenv("ANGLE_ARROW", "weights/angle-models/best_resnet18_arrow.pth"),
# }
# _ANGLE_NUM_CLASSES = {
#     "rectangle": 4,
#     "Triangle": 4,
#     "Racetrack": 4,
#     "Pentagon": 2,
#     "arrow": 4,
# }
# _ANGLE_MODELS = None  
# _ANGLE_TRANSFORM = None

# def _angle_label_for(label: str, idx: int) -> str:
#     try:
#         i = int(idx)
#     except Exception:
#         return ""
#     if label == "Pentagon":
#         return "0 degrees" if i == 0 else ("180 degrees" if i == 1 else "")
#     mapping4 = {0: "0 degrees", 1: "90 degrees", 2: "180 degrees", 3: "270 degrees"}
#     return mapping4.get(i, "")

# def _get_angle_transform():
#     global _ANGLE_TRANSFORM
#     if _ANGLE_TRANSFORM is not None:
#         return _ANGLE_TRANSFORM
#     if not _TORCH_AVAILABLE:
#         return None
#     _ANGLE_TRANSFORM = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])
#     return _ANGLE_TRANSFORM

# def _load_angle_models():
#     global _ANGLE_MODELS
#     if _ANGLE_MODELS is not None:
#         return _ANGLE_MODELS
#     if not _TORCH_AVAILABLE:
#         _ANGLE_MODELS = {}
#         return _ANGLE_MODELS
#     models_dict = {}
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     for label, path in _ANGLE_MODEL_PATHS.items():
#         num_classes = _ANGLE_NUM_CLASSES.get(label)
#         if not num_classes:
#             continue
#         if not os.path.exists(path):
#             continue
#         try:
#             m = models.resnet18(pretrained=False)
#             m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
#             state = torch.load(path, map_location=device)
#             if isinstance(state, dict) and "state_dict" in state:
#                 state = state["state_dict"]
#             try:
#                 m.load_state_dict(state, strict=False)
#             except Exception:
#                 m.load_state_dict(state, strict=False)
#             m.to(device)
#             m.eval()
#             models_dict[label] = (m, device)
#         except Exception:
#             continue
#     _ANGLE_MODELS = models_dict
#     return _ANGLE_MODELS

# def _crop_to_pil(image_bgr, tl, br):
#     import cv2
#     x1, y1 = map(int, map(round, tl))
#     x2, y2 = map(int, map(round, br))
#     h, w = image_bgr.shape[:2]
#     x1, y1 = max(0, x1), max(0, y1)
#     x2, y2 = min(w, x2), min(h, y2)
#     if x2 <= x1 or y2 <= y1:
#         return None
#     crop = image_bgr[y1:y2, x1:x2]
#     if crop.size == 0:
#         return None
#     rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#     return Image.fromarray(rgb)

# def _predict_angle_for_det(image_bgr, label: str, tl, br, models_dict: Dict[str, Any]) -> str:
#     if label not in _ANGLE_ALLOWED:
#         return ""
#     if not models_dict or label not in models_dict:
#         return ""
#     transform = _get_angle_transform()
#     if transform is None:
#         return ""
#     pil_img = _crop_to_pil(image_bgr, tl, br)
#     if pil_img is None:
#         return ""
#     tensor = transform(pil_img).unsqueeze(0)
#     model, device = models_dict[label]
#     try:
#         with torch.no_grad():
#             logits = model(tensor.to(device))
#             idx = int(torch.argmax(logits, dim=1).item())
#         return _angle_label_for(label, idx) or ""
#     except Exception:
#         return ""

# def predict_angles(image_bgr, detections):
#     models_dict = _load_angle_models()
#     if not models_dict:
#         return {}
#     angle_map = {}
#     for det in detections:
#         try:
#             label, tl, br = det
#         except Exception:
#             continue
#         angle = _predict_angle_for_det(image_bgr, label, tl, br, models_dict)
#         angle_map[det] = angle if isinstance(angle, str) else ""
#     return angle_map






import os
import logging
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except Exception:
    torch = None
    nn = None
    models = None
    transforms = None

log = logging.getLogger("angle_detection")

# Backwards-compatible constants expected by sam2_processor
_TORCH_AVAILABLE = True if torch is not None else False

# Angle configuration mirrors notebook
_ANGLE_ALLOWED = {"rectangle", "Triangle", "Racetrack", "Pentagon", "arrow"}
_ANGLE_MODEL_PATHS = {
    "rectangle": os.getenv("ANGLE_RECTANGLE", "weights/angle-models/final_rectangle.pth"),
    "Triangle": os.getenv("ANGLE_TRIANGLE", "weights/angle-models/Triangle.pth"),
    "Racetrack": os.getenv("ANGLE_RACETRACK", "weights/angle-models/final_racetrack.pth"),
    "Pentagon": os.getenv("ANGLE_PENTAGON", "weights/angle-models/best_resnet18_pentagon.pth"),
    "arrow": os.getenv("ANGLE_ARROW", "weights/angle-models/best_resnet18_arrow.pth"),
}
_ANGLE_NUM_CLASSES = {
    "rectangle": 4,
    "Triangle": 4,
    "Racetrack": 4,
    "Pentagon": 2,
    "arrow": 4,
}

_ANGLE_MODELS = None
_ANGLE_TRANSFORM = None


def _angle_label_for(label: str, idx: int) -> str:
    try:
        i = int(idx)
    except Exception:
        return ""
    if label == "Pentagon":
        return "0 degrees" if i == 0 else ("180 degrees" if i == 1 else "")
    mapping4 = {0: "0 degrees", 1: "90 degrees", 2: "180 degrees", 3: "270 degrees"}
    return mapping4.get(i, "")


def _get_angle_transform():
    global _ANGLE_TRANSFORM
    if _ANGLE_TRANSFORM is not None:
        return _ANGLE_TRANSFORM
    if not _TORCH_AVAILABLE:
        return None
    _ANGLE_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return _ANGLE_TRANSFORM


def _load_angle_models():
    global _ANGLE_MODELS
    if _ANGLE_MODELS is not None:
        return _ANGLE_MODELS
    if not _TORCH_AVAILABLE:
        _ANGLE_MODELS = {}
        return _ANGLE_MODELS
    models_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for label, path in _ANGLE_MODEL_PATHS.items():
        num_classes = _ANGLE_NUM_CLASSES.get(label)
        if not num_classes:
            continue
        if not os.path.exists(path):
            continue
        try:
            m = models.resnet18(pretrained=False)
            m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
            state = torch.load(path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            try:
                m.load_state_dict(state, strict=False)
            except Exception:
                m.load_state_dict(state, strict=False)
            m.to(device)
            m.eval()
            models_dict[label] = (m, device)
        except Exception:
            continue
    _ANGLE_MODELS = models_dict
    return _ANGLE_MODELS


def _crop_to_pil(image_bgr, tl, br):
    import cv2
    x1, y1 = map(int, map(round, tl))
    x2, y2 = map(int, map(round, br))
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _predict_angle_for_det(image_bgr, label: str, tl, br, models_dict: dict) -> str:
    if label not in _ANGLE_ALLOWED:
        return ""
    if not models_dict or label not in models_dict:
        return ""
    transform = _get_angle_transform()
    if transform is None:
        return ""
    pil_img = _crop_to_pil(image_bgr, tl, br)
    if pil_img is None:
        return ""
    tensor = transform(pil_img).unsqueeze(0)
    model, device = models_dict[label]
    try:
        with torch.no_grad():
            logits = model(tensor.to(device))
            idx = int(torch.argmax(logits, dim=1).item())
        return _angle_label_for(label, idx) or ""
    except Exception:
        return ""


def predict_angles(image_bgr, detections):
    models_dict = _load_angle_models()
    if not models_dict:
        return {}
    angle_map = {}
    for det in detections:
        try:
            label, tl, br = det[0], det[1], det[2]
        except Exception:
            continue
        angle = _predict_angle_for_det(image_bgr, label, tl, br, models_dict)
        angle_map[det] = angle if isinstance(angle, str) else ""
    return angle_map