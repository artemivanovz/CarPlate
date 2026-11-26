"""
Инференс FastReID SBS ResNet-50-IBN (+Non-local, GeM, BNNeck) без установки fast-reid.

Архитектура повторяет fastreid/modeling (backbones/resnet.py, heads/embedding_head.py),
поэтому чекпоинт veri_sbs_R50-ibn.pth загружается строго, ключ в ключ.
Выход — 2048-d признак после BNNeck, как в режиме eval у FastReID.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class IBN(nn.Module):
    def __init__(self, planes: int) -> None:
        super().__init__()
        self.half = planes // 2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.split(x, self.half, 1)
        return torch.cat((self.IN(a.contiguous()), self.BN(b.contiguous())), 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, with_ibn: bool, stride: int = 1,
                 downsample: nn.Module = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = IBN(planes) if with_ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class NonLocal(nn.Module):
    # В FastReID inter_channels = reduc_ratio // reduc_ratio = 1; веса обучены именно так.
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.g = nn.Conv2d(in_channels, 1, 1)
        self.W = nn.Sequential(nn.Conv2d(1, in_channels, 1), nn.BatchNorm2d(in_channels))
        self.theta = nn.Conv2d(in_channels, 1, 1)
        self.phi = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        g_x = self.g(x).view(n, 1, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(n, 1, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(n, 1, -1)
        f = torch.matmul(theta_x, phi_x)
        y = torch.matmul(f / f.size(-1), g_x).permute(0, 2, 1).contiguous()
        return self.W(y.view(n, 1, *x.shape[2:])) + x


class ResNetIBNNL(nn.Module):
    layers = (3, 4, 6, 3)
    non_layers = (0, 2, 3, 0)

    def __init__(self, last_stride: int = 1) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(64, self.layers[0], 1, True)
        self.layer2 = self._make_layer(128, self.layers[1], 2, True)
        self.layer3 = self._make_layer(256, self.layers[2], 2, True)
        self.layer4 = self._make_layer(512, self.layers[3], last_stride, False)
        self.NL_2 = nn.ModuleList([NonLocal(512) for _ in range(self.non_layers[1])])
        self.NL_3 = nn.ModuleList([NonLocal(1024) for _ in range(self.non_layers[2])])
        # Non-local вставляется после последних блоков слоя: для layer2 — после 2 и 3, layer3 — после 3, 4, 5.
        self.nl_idx = {
            2: sorted(self.layers[1] - (i + 1) for i in range(self.non_layers[1])),
            3: sorted(self.layers[2] - (i + 1) for i in range(self.non_layers[2])),
        }

    def _make_layer(self, planes: int, blocks: int, stride: int, with_ibn: bool) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [Bottleneck(self.inplanes, planes, with_ibn, stride, downsample)]
        self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes, with_ibn) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        for stage, layer, nl in ((2, self.layer2, self.NL_2), (3, self.layer3, self.NL_3)):
            idx = self.nl_idx[stage]
            for i, block in enumerate(layer):
                x = block(x)
                if i in idx:
                    x = nl[idx.index(i)](x)
        return self.layer4(x)


class GeMP(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class EmbeddingHead(nn.Module):
    def __init__(self, feat_dim: int = 2048) -> None:
        super().__init__()
        self.pool_layer = GeMP()
        self.bottleneck = nn.Sequential(nn.BatchNorm2d(feat_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.pool_layer(x))[..., 0, 0]


class FastReIDSBS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNetIBNNL(last_stride=1)
        self.heads = EmbeddingHead()
        self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def forward(self, images_0_255: torch.Tensor) -> torch.Tensor:
        return self.heads(self.backbone((images_0_255 - self.pixel_mean) / self.pixel_std))


def load_fastreid(path: str, device: str) -> FastReIDSBS:
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    # Классификатор обучающих ID (575 машин VeRi) и счётчик старого BNNeck на инференсе не нужны.
    state = {k: v for k, v in state.items()
             if not k.startswith("heads.classifier") and k != "heads.bnneck.num_batches_tracked"}
    model = FastReIDSBS()
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def fastreid_embed_fn(path: str, device: str, input_size=(256, 256)):
    model = load_fastreid(path, device)
    h, w = input_size

    @torch.no_grad()
    def run(rgb: List[np.ndarray]) -> np.ndarray:
        # FastReID на тесте: RGB, bicubic-resize, значения 0..255 без деления.
        batch = np.stack([cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC) for im in rgb])
        x = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float()
        return model(x).cpu().numpy()

    return run
