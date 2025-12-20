"""Utility functions for face anti-spoofing inference."""

import cv2
import numpy as np
import torch


def xyxy2xywh(bbox: np.ndarray | list) -> np.ndarray:
    """Convert bounding box from [x1, y1, x2, y2] to [x, y, w, h] format.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format.

    Returns:
        Bounding box in [x, y, w, h] format.
    """
    if isinstance(bbox, list):
        bbox = np.array(bbox)

    result = np.copy(bbox)
    result[..., 2] = bbox[..., 2] - bbox[..., 0]
    result[..., 3] = bbox[..., 3] - bbox[..., 1]
    return result


def crop_face(
    image: np.ndarray,
    bbox: list[int],
    scale: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Crop and resize face region from image.

    Args:
        image: Input image (BGR format).
        bbox: Bounding box [x, y, w, h].
        scale: Scale factor for expanding crop region.
        out_w: Output width.
        out_h: Output height.

    Returns:
        Cropped and resized face image.
    """
    src_h, src_w = image.shape[:2]
    x, y, box_w, box_h = bbox

    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w = box_w * scale
    new_h = box_h * scale

    center_x = x + box_w / 2
    center_y = y + box_h / 2

    x1 = max(0, int(center_x - new_w / 2))
    y1 = max(0, int(center_y - new_h / 2))
    x2 = min(src_w - 1, int(center_x + new_w / 2))
    y2 = min(src_h - 1, int(center_y + new_h / 2))

    cropped = image[y1 : y2 + 1, x1 : x2 + 1]
    return cv2.resize(cropped, (out_w, out_h))


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert numpy image (HWC) to PyTorch tensor (CHW).

    Args:
        image: Image array in HWC format.

    Returns:
        Float tensor in CHW format.
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def draw_bbox(
    image: np.ndarray,
    bbox: list[int],
    label: str,
    score: float,
    color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Draw bounding box with label on image.

    Args:
        image: Input image (will be modified in-place).
        bbox: Bounding box [x, y, w, h].
        label: Text label ("Real" or "Fake").
        score: Confidence score.
        color: BGR color tuple. If None, green for Real, red for Fake.

    Returns:
        Image with drawn bounding box.
    """
    x, y, w, h = bbox
    if color is None:
        color = (0, 255, 0) if label == "Real" else (0, 0, 255)

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = f"{label}: {score:.2f}"
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image
