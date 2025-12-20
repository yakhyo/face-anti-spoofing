"""Export MiniFASNet models to ONNX format."""

import argparse
from pathlib import Path

import torch

from models import MiniFASNetV1SE, MiniFASNetV2

MODEL_REGISTRY = {
    "v1se": MiniFASNetV1SE,
    "v2": MiniFASNetV2,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MiniFASNet models to ONNX")
    parser.add_argument("-w", "--weight", type=str, required=True, help="Path to .pth weight file")
    parser.add_argument(
        "-n",
        "--model",
        type=str,
        default="v1se",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model variant",
    )
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch size")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output ONNX filename")
    return parser.parse_args()


@torch.no_grad()
def export_onnx(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls()
    model.to(device)

    state_dict = torch.load(args.weight, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {args.weight}")

    model.eval()

    if args.output:
        onnx_path = args.output
    else:
        weight_path = Path(args.weight)
        onnx_path = str(weight_path.with_suffix(".onnx"))

    print(f"Exporting model to '{onnx_path}'")

    dummy_input = torch.randn(1, 3, 80, 80).to(device)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
        print("Using dynamic batch size")
    else:
        print("Using fixed input size: (1, 3, 80, 80)")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    print(f"✓ Model exported successfully to {onnx_path}")


if __name__ == "__main__":
    args = parse_arguments()
    export_onnx(args)
