"""Face anti-spoofing detection using MiniFASNet models."""

import argparse
import warnings

import cv2
import numpy as np
import torch
from uniface import RetinaFace

from models import MiniFASNetV1SE, MiniFASNetV2
from utils import crop_face, draw_bbox, to_tensor, xyxy2xywh

warnings.filterwarnings("ignore")

MODEL_CONFIGS = {
    "v1se": {"class": MiniFASNetV1SE, "input_size": (80, 80), "scale": 4.0},
    "v2": {"class": MiniFASNetV2, "input_size": (80, 80), "scale": 2.7},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Face Anti-Spoofing Detection")

    # Model arguments
    parser.add_argument("--weight", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--model", type=str, default="v2", choices=["v1se", "v2"], help="Model variant")

    # Input/Output arguments
    parser.add_argument("--source", type=str, default="0", help="Image path or camera index")
    parser.add_argument("--output", type=str, default=None, help="Path to save output (image or video)")
    parser.add_argument("--view", action="store_true", help="Display inference results")

    # Processing arguments
    parser.add_argument("--confidence", type=float, default=0.5, help="Face detection confidence threshold")

    return parser.parse_args()


def load_model(weight_path: str, model_name: str, device: torch.device):
    """Load anti-spoofing model."""
    config = MODEL_CONFIGS[model_name]
    model = config["class"]()
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.to(device).eval()
    print(f"Loaded {model_name} model on {device}")
    return model, config


def predict(image: np.ndarray, bbox: list, model, config: dict, device: torch.device) -> dict:
    """Predict if a face is real or fake."""
    bbox_xywh = xyxy2xywh(bbox).astype(int).tolist()
    h, w = config["input_size"]

    face_crop = crop_face(image, bbox_xywh, config["scale"], w, h)
    tensor = to_tensor(face_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()

    label_idx = int(np.argmax(probs))
    score = float(probs[0, label_idx])

    return {
        "label": "Real" if label_idx == 1 else "Fake",
        "score": score,
        "bbox": bbox_xywh,
    }


def run_image(args, model, config, detector, device):
    """Process a single image."""
    image = cv2.imread(args.source)
    if image is None:
        raise ValueError(f"Failed to load image: {args.source}")

    faces = detector.detect(image)
    faces = [f for f in faces if f["confidence"] >= args.confidence]

    if not faces:
        print("No faces detected.")
        return

    print(f"Found {len(faces)} face(s)")
    for i, face in enumerate(faces):
        result = predict(image, face["bbox"], model, config, device)
        draw_bbox(image, result["bbox"], result["label"], result["score"])
        print(f"  Face {i + 1}: {result['label']} (score={result['score']:.2f})")

    if args.output:
        cv2.imwrite(args.output, image)
        print(f"Saved: {args.output}")

    if args.view:
        cv2.imshow("Anti-Spoofing", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam(args, model, config, detector, device):
    """Process webcam stream."""
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else 0)
    if not cap.isOpened():
        raise ValueError("Failed to open camera")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("Running webcam inference... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)
        for face in faces:
            if face["confidence"] < args.confidence:
                continue
            result = predict(frame, face["bbox"], model, config, device)
            draw_bbox(frame, result["bbox"], result["label"], result["score"])

        if writer:
            writer.write(frame)

        if args.view:
            cv2.imshow("Anti-Spoofing", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {args.output}")
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    if not args.view and not args.output:
        raise ValueError("At least one of --view or --output must be provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.weight, args.model, device)
    detector = RetinaFace()

    # Determine if source is image or webcam
    is_image = args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    if is_image:
        run_image(args, model, config, detector, device)
    else:
        run_webcam(args, model, config, detector, device)


if __name__ == "__main__":
    main()
