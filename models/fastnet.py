"""MiniFASNet: Lightweight Face Anti-Spoofing Network."""

import torch
import torch.nn as nn

CHANNEL_CONFIGS = {
    "v1se": {
        "stem": 32,
        "transition1": {"expand": 103, "out": 64},
        "stage2": [(13, 64), (26, 64), (13, 64), (52, 64)],
        "transition2": {"expand": 231, "out": 128},
        "stage3": [(154, 128), (52, 128), (26, 128), (52, 128), (26, 128), (26, 128)],
        "transition3": {"expand": 308, "out": 128},
        "stage4": [(26, 128), (26, 128)],
        "final": 512,
    },
    "v2": {
        "stem": 32,
        "transition1": {"expand": 103, "out": 64},
        "stage2": [(13, 64), (13, 64), (13, 64), (13, 64)],
        "transition2": {"expand": 231, "out": 128},
        "stage3": [(231, 128), (52, 128), (26, 128), (77, 128), (26, 128), (26, 128)],
        "transition3": {"expand": 308, "out": 128},
        "stage4": [(26, 128), (26, 128)],
        "final": 512,
    },
}


class ConvBNPReLU(nn.Module):
    """Conv2d + BatchNorm2d + PReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 1,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation channel attention module."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.relu(self.bn1(self.fc1(scale)))
        scale = self.sigmoid(self.bn2(self.fc2(scale)))
        return x * scale


class InvertedResidual(nn.Module):
    """Inverted residual block: expand -> depthwise -> project (+ optional SE & residual)."""

    def __init__(
        self,
        in_channels: int,
        expand_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        self.conv = ConvBNPReLU(in_channels, expand_channels, kernel_size=1)
        self.conv_dw = ConvBNPReLU(expand_channels, expand_channels, 3, stride, 1, groups=expand_channels)
        self.project = ConvBNPReLU(expand_channels, out_channels, kernel_size=1, activation=False)
        self.se = SqueezeExcite(out_channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.conv_dw(out)
        out = self.project(out)
        out = self.se(out)

        if self.use_residual:
            out = out + x
        return out


class ResidualStack(nn.Module):
    """Stack of inverted residual blocks (SE applied to last block only)."""

    def __init__(
        self,
        in_channels: int,
        block_configs: list[tuple[int, int]],
        use_se: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        current_ch = in_channels

        for i, (expand_ch, out_ch) in enumerate(block_configs):
            layers.append(
                InvertedResidual(
                    in_channels=current_ch,
                    expand_channels=expand_ch,
                    out_channels=out_ch,
                    stride=1,
                    use_se=use_se and (i == len(block_configs) - 1),
                )
            )
            current_ch = out_ch

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MiniFASNet(nn.Module):
    """Lightweight Face Anti-Spoofing Network (expects 80×80 input)."""

    def __init__(
        self,
        config: dict,
        dropout: float = 0.0,
        num_classes: int = 3,
        use_se: bool = False,
    ) -> None:
        super().__init__()

        stem_ch = config["stem"]
        t1 = config["transition1"]
        t2 = config["transition2"]
        t3 = config["transition3"]
        final_ch = config["final"]

        self.stem = ConvBNPReLU(3, stem_ch, kernel_size=3, stride=2, padding=1)
        self.stem_dw = ConvBNPReLU(stem_ch, stem_ch, kernel_size=3, stride=1, padding=1, groups=stem_ch)

        self.transition1 = InvertedResidual(stem_ch, t1["expand"], t1["out"], stride=2)
        self.stage2 = ResidualStack(t1["out"], config["stage2"], use_se=use_se)

        self.transition2 = InvertedResidual(config["stage2"][-1][1], t2["expand"], t2["out"], stride=2)
        self.stage3 = ResidualStack(t2["out"], config["stage3"], use_se=use_se)

        self.transition3 = InvertedResidual(config["stage3"][-1][1], t3["expand"], t3["out"], stride=2)
        self.stage4 = ResidualStack(t3["out"], config["stage4"], use_se=use_se)

        stage4_out = config["stage4"][-1][1]
        self.final_expand = ConvBNPReLU(stage4_out, final_ch, kernel_size=1)
        self.final_dw = ConvBNPReLU(final_ch, final_ch, kernel_size=5, groups=final_ch, activation=False)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(final_ch, 128, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(128, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stem_dw(x)
        x = self.transition1(x)
        x = self.stage2(x)
        x = self.transition2(x)
        x = self.stage3(x)
        x = self.transition3(x)
        x = self.stage4(x)
        x = self.final_expand(x)
        x = self.final_dw(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.drop(x)
        x = self.classifier(x)
        return x


class MiniFASNetV1SE(MiniFASNet):
    """MiniFASNet V1 with Squeeze-and-Excitation modules (~0.43M params)."""

    def __init__(self) -> None:
        super().__init__(config=CHANNEL_CONFIGS["v1se"], dropout=0.75, num_classes=3, use_se=True)


class MiniFASNetV2(MiniFASNet):
    """MiniFASNet V2 without SE modules for faster inference (~0.43M params)."""

    def __init__(self) -> None:
        super().__init__(config=CHANNEL_CONFIGS["v2"], dropout=0.2, num_classes=3, use_se=False)
