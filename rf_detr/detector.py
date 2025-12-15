from typing import List, Tuple, Union

from PIL import Image


_RFDETR_MODEL = None
_CLASS_NAMES = [
    "objects",
    "Cloud",
    "Diamond",
    "Double Arrow",
    "Pentagon",
    "Racetrack",
    "Star",
    "Sticky Notes",
    "Triangle",
    "arrow",
    "arrow_head",
    "circle",
    "dashed-arrow",
    "dotted-arrow",
    "rectangle",
    "rounded rectangle",
    "solid-arrow",
]

# Type for detection: (label, top_left, bottom_right, confidence)
ShapeDetection = Tuple[str, Tuple[float, float], Tuple[float, float], float]


def _load_model(weights_path: str = "weights/pre-trained-model/checkpoint_best_regular.pth"):
    global _RFDETR_MODEL
    if _RFDETR_MODEL is not None:
        return _RFDETR_MODEL
    try:
        from rfdetr import RFDETRMedium
        model = RFDETRMedium(pretrain_weights=weights_path)
        model.optimize_for_inference()
        _RFDETR_MODEL = model
        return _RFDETR_MODEL
    except Exception as exc:  
        print(f"[RF-DETR] Failed to load model: {exc}")
        _RFDETR_MODEL = None
        return None


def detect_shapes(file_path: str) -> List[ShapeDetection]:
   
    """Run RF-DETR inference and return list of (label, (x1,y1), (x2,y2), confidence).

    Returns empty list if model or image cannot be loaded.
    """
    model = _load_model()
    if model is None:
        return []

    try:
        image = Image.open(file_path).convert("RGB")
    except Exception as exc:
        print(f"[RF-DETR] Failed to open image: {exc}")
        return []

    try:
        detections = model.predict(image, threshold=0.60)
        results: List[ShapeDetection] = []
        for bbox, class_id, conf in zip(detections.xyxy, detections.class_id, detections.confidence):
            x_min, y_min, x_max, y_max = bbox
            class_name = _CLASS_NAMES[int(class_id)] if 0 <= int(class_id) < len(_CLASS_NAMES) else str(class_id)
            results.append(
                (
                    class_name,
                    (float(x_min), float(y_min)),
                    (float(x_max), float(y_max)),
                    float(conf),
                )
            )
        return results
    except Exception as exc:  
        print(f"[RF-DETR] Inference failed: {exc}")
        return []

