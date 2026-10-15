"""Paper student architectures with NCHW inputs and unnormalized features."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SameConv2d(nn.Conv2d):
    """TensorFlow SAME padding, including even kernels and strided convolutions."""

    def forward(self, x: Tensor) -> Tensor:
        padding = []
        for size, kernel, stride, dilation in zip(
            reversed(x.shape[-2:]), reversed(self.kernel_size),
            reversed(self.stride), reversed(self.dilation),
        ):
            total = max((math.ceil(size / stride) - 1) * stride + dilation * (kernel - 1) + 1 - size, 0)
            padding.extend((total // 2, total - total // 2))
        return F.conv2d(F.pad(x, padding), self.weight, self.bias, self.stride,
                        0, self.dilation, self.groups)


def _conv(inputs, outputs, kernel, *, stride=1, bias=False, regularize=True):
    layer = SameConv2d(inputs, outputs, kernel, stride=stride, bias=bias)
    layer.regularize_kernel = regularize
    return layer


def _batch_norm(channels):
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=.01)


def _glorot(layer):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def _he_normal(layer):
    if isinstance(layer, nn.Conv2d):
        # Keras HeNormal uses a variance-corrected, two-sigma truncated normal.
        std = math.sqrt(2 / (layer.in_channels * math.prod(layer.kernel_size))) / .8796256610342398
        nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    elif isinstance(layer, nn.Linear):
        _glorot(layer)


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.conv1 = _conv(inputs, outputs, 3, stride=stride)
        self.bn1 = _batch_norm(outputs)
        self.conv2 = _conv(outputs, outputs, 3)
        self.bn2 = _batch_norm(outputs)
        self.stride, self.extra_channels = stride, outputs - inputs

    def forward(self, x):
        residual = x[:, :, ::self.stride, ::self.stride]
        if self.extra_channels:
            residual = F.pad(residual, (0, 0, 0, 0, 0, self.extra_channels))
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)) + residual)


class ResNet20(nn.Module):
    """CIFAR ResNet-20 with original blocks and option-A shortcuts."""

    feature_dim = 64
    l2_coefficient = 1e-4

    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(_conv(in_channels, 16, 3), _batch_norm(16), nn.ReLU())
        blocks, channels = [], 16
        for width, stride in ((16, 1), (32, 2), (64, 2)):
            for index in range(3):
                blocks.append(ResidualBlock(channels, width, stride if index == 0 else 1))
                channels = width
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.classifier.regularize_kernel = True
        self.apply(_he_normal)

    def forward(self, x, return_features=False):
        features = self.blocks(self.stem(x)).mean(dim=(-2, -1))
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


class AudioBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.frequency = nn.Sequential(_conv(inputs, outputs, (1, 4), bias=True),
                                      nn.GroupNorm(4, outputs, eps=1e-3), nn.ReLU())
        self.temporal = nn.Sequential(_conv(inputs, outputs, (4, 1), bias=True),
                                       nn.GroupNorm(4, outputs, eps=1e-3), nn.ReLU())
        self.joint = nn.Sequential(_conv(2 * outputs, outputs, 1, bias=True),
                                  nn.GroupNorm(4, outputs, eps=1e-3), nn.ReLU(),
                                  nn.MaxPool2d(2), nn.Dropout2d(.1))

    def forward(self, x):
        return self.joint(torch.cat((self.frequency(x), self.temporal(x)), dim=1))


class AudioCNN(nn.Module):
    """Four time/frequency convolution blocks from FedLN Section 5.2."""

    feature_dim = 128
    l2_coefficient = 1e-4

    def __init__(self, num_classes=12):
        super().__init__()
        blocks, channels = [], 1
        for width in (24, 32, 64, 128):
            blocks.append(AudioBlock(channels, width))
            channels = width
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.apply(_glorot)

    def forward(self, x, return_features=False):
        features = self.blocks(x).amax(dim=(-2, -1))
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


class MixerBlock(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.depthwise = SameConv2d(width, width, 2, groups=width)
        self.spatial = nn.Sequential(nn.GELU(), nn.BatchNorm2d(width))
        self.channel = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.BatchNorm2d(width))

    def forward(self, x):
        return self.channel(x + self.spatial(self.depthwise(x)))


class ConvMixer(nn.Module):
    """ConvMixer-128/4 with kernel 2 and patch 5 (FedLN Section 5.3)."""

    feature_dim = 128
    l2_coefficient = 0.

    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 128, 5, stride=5),
                                  nn.GELU(), nn.BatchNorm2d(128))
        self.blocks = nn.Sequential(*(MixerBlock() for _ in range(4)))
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, return_features=False):
        features = self.blocks(self.stem(x)).mean(dim=(-2, -1))
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


_CLASSES = {"cifar10": 10, "cifar100": 100, "cifar10n": 10, "cifar100n": 100,
            "fashionmnist": 10, "pathmnist": 9, "eurosat": 10, "spcm": 12}


def build_model(dataset: str, model: str = "auto") -> nn.Module:
    if dataset not in _CLASSES:
        raise ValueError(f"Unknown dataset: {dataset}")
    if model == "auto":
        model = "cnn" if dataset == "spcm" else "resnet20"
    if dataset == "spcm":
        if model != "cnn":
            raise ValueError("Speech Commands uses the audio CNN; select --model cnn or auto")
        return AudioCNN()
    inputs = 1 if dataset == "fashionmnist" else 3
    constructors = {"resnet20": ResNet20, "convmixer": ConvMixer}
    if model not in constructors:
        raise ValueError(f"Unknown vision model: {model}")
    return constructors[model](_CLASSES[dataset], inputs)


def regularization_loss(model: nn.Module) -> Tensor:
    """Explicit kernel L2; optimizer weight decay must remain zero."""
    total = next(model.parameters()).new_zeros(())
    coefficient = getattr(model, "l2_coefficient", 0.)
    if coefficient:
        for layer in model.modules():
            if getattr(layer, "regularize_kernel", False):
                total = total + layer.weight.square().sum()
    return total * coefficient
