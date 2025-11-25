import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


# 1. CIFAR-10 model (DenseNet-169)

class CifarClientModel(nn.Module):
    """
    Client model: DenseNet-169 adapted for CIFAR-10 (32x32).

    - Replace first conv with 3x3, stride=1, padding=1
    - Remove initial pooling
    - Keep 32x32 resolution much longer → cheaper than 224x224
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        try:
            base = tv_models.densenet169(weights=None)
        except TypeError:
            base = tv_models.densenet169(pretrained=None)

        # CIFAR-friendly stem
        base.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        base.features.pool0 = nn.Identity()

        in_feats = base.classifier.in_features
        base.classifier = nn.Linear(in_feats, num_classes)
        self.base = base

    def forward(self, x):
        return self.base(x)




# 2. UTMobileNet traffic model (ResNet1D)

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # 5x1 conv for 1D data
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5, stride=1, padding=2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm1d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for UTMobileNet traffic features.

    Args:
        num_features: number of input features per packet/flow
        num_classes:  number of application classes
    """
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()

        self.in_channels = 64
        # initial 1D conv over features
        self.initial = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * Bottleneck1D.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(Bottleneck1D(self.in_channels, out_channels, s))
            self.in_channels = out_channels * Bottleneck1D.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, num_features]
        x = x.unsqueeze(1)           # [B, 1, F]
        x = F.relu(self.initial(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)         # [B, C, 1]
        x = x.view(x.size(0), -1)    # [B, C]
        x = self.fc(x)
        return x




def build_model(dataset: str, **kwargs) -> nn.Module:
    """
    Simple factory to pick the right model.

    Examples:
        build_model("cifar10", num_classes=10)
        build_model("utmobilenet", num_features=21, num_classes=14)
    """
    dataset = dataset.lower()
    if dataset == "cifar10":
        num_classes = kwargs.get("num_classes", 10)
        return CifarClientModel(num_classes=num_classes)
    elif dataset in ["utmobilenet", "traffic"]:
        num_features = kwargs["num_features"]    # required
        num_classes = kwargs["num_classes"]      # required
        return ResNet1D(num_features=num_features, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown dataset '{dataset}' for build_model()")

