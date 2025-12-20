"""Face anti-spoofing detection using MiniFASNet models."""

import argparse
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from uniface import RetinaFace

from models import MiniFASNetV1SE, MiniFASNetV2
from utils import crop_face, draw_bbox, to_tensor, xyxy2xywh

warnings.filterwarnings("ignore")

MODEL_CONFIGS: dict[str, dict] = {
    "MiniFASNetV2.pth": {
        "class": MiniFASNetV2,
        "input_size": (80, 80),
        "scale": 2.7,
    },
    "MiniFASNetV1SE.pth": {
        "class": MiniFASNetV1SE,
        "input_size": (80, 80),
        "scale": 4.0,
    },
}


def load_models(
    model_dir: str | Path,
    device: torch.device | None = None,
) -> dict[str, tuple[nn.Module, dict]]:
    """Load all anti-spoofing models from directory.

    Args:
        model_dir: Directory containing .pth model files.
        device: Target device. If None, auto-selects CUDA if available.

    Returns:
        Dictionary mapping model name to (model, config) tuple.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(model_dir)
    models = {}

    for model_file in model_dir.glob("*.pth"):
        name = model_file.name
        if name not in MODEL_CONFIGS:
            print(f"Warning: Unknown model {name}, skipping...")
            continue

        config = MODEL_CONFIGS[name]
        model = config["class"]()
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        models[name] = (model, config)

    if not models:
        raise ValueError(f"No valid models found in {model_dir}")

    print(f"Loaded {len(models)} model(s) on {device}")
    return models


def predict(
    image: np.ndarray,
    bbox_xyxy: list[float],
    models: dict[str, tuple[nn.Module, dict]],
    device: torch.device,
) -> dict:
    """Predict if a face is real or fake using ensemble of models.

    Args:
        image: Input image (BGR format).
        bbox_xyxy: Face bounding box [x1, y1, x2, y2].
        models: Dictionary from load_models().
        device: Target device.

    Returns:
        Dictionary with keys: label, score, bbox (xywh format), time.
    """
    bbox_xywh = xyxy2xywh(bbox_xyxy).astype(int).tolist()
    predictions = np.zeros((1, 3))
    total_time = 0.0

    for _, (model, config) in models.items():
        h, w = config["input_size"]
        scale = config["scale"]

        face_crop = crop_face(image, bbox_xywh, scale, w, h)
        tensor = to_tensor(face_crop).unsqueeze(0).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()
        total_time += time.perf_counter() - start

        predictions += probs

    label_idx = int(np.argmax(predictions))
    score = predictions[0, label_idx] / len(models)

    return {
        "label": "Real" if label_idx == 1 else "Fake",
        "score": float(score),
        "bbox": bbox_xywh,
        "time": total_time,
    }


def process_image(
    image_path: str | Path,
    models: dict[str, tuple[nn.Module, dict]],
    detector: RetinaFace,
    device: torch.device,
    confidence_threshold: float = 0.5,
) -> list[dict]:
    """Process an image and detect spoofing for all faces.

    Args:
        image_path: Path to input image.
        models: Dictionary from load_models().
        detector: RetinaFace detector instance.
        device: Target device.
        confidence_threshold: Minimum face detection confidence.

    Returns:
        List of prediction results for each detected face.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    faces = detector.detect(image)
    faces = [f for f in faces if f["confidence"] >= confidence_threshold]

    if not faces:
        print("No faces detected.")
        return []

    print(f"Found {len(faces)} face(s)")

    results = []
    for i, face in enumerate(faces):
        result = predict(image, face["bbox"], models, device)
        result["detection_confidence"] = float(face["confidence"])
        results.append(result)
        print(f"  Face {i + 1}: {result['label']} (score={result['score']:.2f}, time={result['time'] * 1000:.1f}ms)")

    return results


def process_webcam(
    camera_idx: int,
    models: dict[str, tuple[nn.Module, dict]],
    detector: RetinaFace,
    device: torch.device,
    confidence_threshold: float = 0.5,
) -> None:
    """Process webcam stream and detect spoofing in real-time.

    Args:
        camera_idx: Camera index (0 for default webcam).
        models: Dictionary from load_models().
        detector: RetinaFace detector instance.
        device: Target device.
        confidence_threshold: Minimum face detection confidence.
    """
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise ValueError(f"Failed to open camera: {camera_idx}")

    print("Running webcam inference... Press 'q' to quit")
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            faces = detector.detect(frame)
            for face in faces:
                if face["confidence"] < confidence_threshold:
                    continue

                result = predict(frame, face["bbox"], models, device)
                draw_bbox(frame, result["bbox"], result["label"], result["score"])

            cv2.imshow("Anti-Spoofing", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames")


def main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="Face Anti-Spoofing Detection")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image path or camera index")
    parser.add_argument("--weights", "-w", type=str, default="./weights", help="Model weights directory")
    parser.add_argument("--device", "-d", type=int, default=0, help="GPU device ID")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="Face detection confidence threshold")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="image",
        choices=["image", "webcam"],
        help="Processing mode",
    )

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    models = load_models(args.weights, device)
    detector = RetinaFace()

    if args.mode == "image":
        results = process_image(args.input, models, detector, device, args.confidence)
        print("\n" + "=" * 50)
        print("Summary:")
        for i, r in enumerate(results):
            print(f"  Face {i + 1}: {r['label']} (score={r['score']:.2f}, conf={r['detection_confidence']:.2f})")

    elif args.mode == "webcam":
        camera_idx = int(args.input) if args.input.isdigit() else 0
        process_webcam(camera_idx, models, detector, device, args.confidence)


if __name__ == "__main__":
    main()
