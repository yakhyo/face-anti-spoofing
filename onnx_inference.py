"""ONNX inference for MiniFASNet face anti-spoofing."""

import argparse

import cv2
import numpy as np
import onnxruntime as ort
from uniface import RetinaFace

from utils import draw_bbox


class AntiSpoofingONNX:
    """Face anti-spoofing inference using ONNXRuntime."""

    def __init__(self, model_path: str, scale: float = 2.7) -> None:
        """Initialize the AntiSpoofingONNX class.

        Args:
            model_path: Path to the ONNX model file.
            scale: Crop scale factor for face region.
        """
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.scale = scale

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_size = tuple(input_cfg.shape[2:])

        output_cfg = self.session.get_outputs()[0]
        self.output_name = output_cfg.name

    def _xyxy2xywh(self, bbox: list[float]) -> list[int]:
        """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
        x1, y1, x2, y2 = bbox
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    def _crop_face(self, image: np.ndarray, bbox: list[int]) -> np.ndarray:
        """Crop and resize face region from image."""
        src_h, src_w = image.shape[:2]
        x, y, box_w, box_h = bbox

        scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, self.scale)
        new_w = box_w * scale
        new_h = box_h * scale

        center_x = x + box_w / 2
        center_y = y + box_h / 2

        x1 = max(0, int(center_x - new_w / 2))
        y1 = max(0, int(center_y - new_h / 2))
        x2 = min(src_w - 1, int(center_x + new_w / 2))
        y2 = min(src_h - 1, int(center_y + new_h / 2))

        cropped = image[y1 : y2 + 1, x1 : x2 + 1]
        return cv2.resize(cropped, self.input_size[::-1])

    def _preprocess(self, image: np.ndarray, bbox: list[int]) -> np.ndarray:
        """Preprocess face crop for inference."""
        face = self._crop_face(image, bbox)
        face = face.astype(np.float32)
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        return face

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def predict(self, image: np.ndarray, bbox_xyxy: list[float]) -> dict:
        """Predict if face is real or fake.

        Args:
            image: Input image (BGR format).
            bbox_xyxy: Face bounding box [x1, y1, x2, y2].

        Returns:
            Dictionary with keys: label, score, bbox (xywh format).
        """
        bbox_xywh = self._xyxy2xywh(bbox_xyxy)

        input_tensor = self._preprocess(image, bbox_xywh)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})

        logits = outputs[0]
        probs = self._softmax(logits)

        label_idx = int(np.argmax(probs))
        score = float(probs[0, label_idx])

        return {
            "label": "Real" if label_idx == 1 else "Fake",
            "score": score,
            "bbox": bbox_xywh,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face Anti-Spoofing ONNX Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    parser.add_argument("--scale", type=float, default=2.7, help="Crop scale (2.7 for V2, 4.0 for V1SE)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise IOError(f"Failed to open camera: {args.source}")

    engine = AntiSpoofingONNX(model_path=args.model, scale=args.scale)
    detector = RetinaFace()

    print("Running webcam inference... Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        for face in faces:
            result = engine.predict(frame, face["bbox"])
            draw_bbox(frame, result["bbox"], result["label"], result["score"])

        cv2.imshow("Anti-Spoofing", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
