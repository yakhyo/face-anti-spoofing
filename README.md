# Face Anti-Spoofing

Minimal inference utilities for silent face anti-spoofing using **MiniFASNetV1SE** and **MiniFASNetV2** models.

## Features

- PyTorch and ONNX Runtime inference
- Real-time webcam detection
- Image processing
- Ensemble prediction support
- Lightweight models (~0.43M parameters)

## Installation

Python 3.10+

```bash
pip install -r requirements.txt
```

## Quick Start

### Image Inference

```bash
python main.py --mode image --input path/to/image.jpg
```

### Webcam Inference

```bash
python main.py --mode webcam --input 0
```

### Options

- `--mode`: Processing mode (`image` or `webcam`)
- `--input`: Image path or camera index
- `--weights`: Model weights directory (default: `./weights`)
- `--device`: GPU device ID (default: 0)
- `--confidence`: Face detection confidence threshold (default: 0.5)

## ONNX Export

Export PyTorch model to ONNX format:

```bash
python scripts/onnx_export.py --model v2 --weight ./weights/MiniFASNetV2.pth
```

Options:

- `--model`: Model variant (`v1se` or `v2`)
- `--weight`: Path to `.pth` weight file
- `--output`: Output ONNX filename (optional)
- `--dynamic`: Enable dynamic batch size (optional)

## ONNX Inference

Run ONNX inference with webcam:

```bash
python scripts/onnx_inference.py --model ./weights/MiniFASNetV2.onnx --scale 2.7
```

Options:

- `--model`: Path to ONNX model file
- `--source`: Camera index (default: 0)
- `--scale`: Crop scale factor (2.7 for V2, 4.0 for V1SE)

## Model Details

| Model          | Parameters | Crop Scale | Input Size |
| -------------- | ---------- | ---------- | ---------- |
| MiniFASNetV1SE | ~0.43M     | 4.0        | 80×80     |
| MiniFASNetV2   | ~0.43M     | 2.7        | 80×80     |

## Reference

Based on [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
