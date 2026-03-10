"""
ResNet backbone variants for SelectiveNet.

Supports ResNet-18 and WideResNet-28-10 as alternative backbones
to VGG for the ablation study (backbone dependence).
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetFeatures(nn.Module):
    """
    ResNet feature extractor that outputs a fixed-dim feature vector.
    
    Wraps torchvision ResNet, removing the final FC layer.
    Output dimension: 512 for ResNet-18, 2048 for ResNet-50.
    """
    
    def __init__(self, arch='resnet18', input_size=32, pretrained=False):
        super().__init__()
        
        if arch == 'resnet18':
            base = models.resnet18(weights=None)
            self.feature_dim = 512
        elif arch == 'resnet50':
            base = models.resnet50(weights=None)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported arch: {arch}")
        
        # For small images (32x32), replace first conv and remove maxpool
        if input_size <= 64:
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()
        
        # Remove the final FC layer
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class WideResNetBlock(nn.Module):
    """Basic wide residual block."""
    
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )
    
    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNetFeatures(nn.Module):
    """
    WideResNet-28-10 feature extractor.
    
    Output dimension: 640 (64 * widen_factor with widen_factor=10).
    Standard architecture for CIFAR benchmarks.
    """
    
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, input_size=32):
        super().__init__()
        
        assert (depth - 4) % 6 == 0, 'WideResNet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        
        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.feature_dim = nStages[3]
        
        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        self.layer1 = self._make_layer(n, nStages[0], nStages[1], dropout_rate, stride=1)
        self.layer2 = self._make_layer(n, nStages[1], nStages[2], dropout_rate, stride=2)
        self.layer3 = self._make_layer(n, nStages[2], nStages[3], dropout_rate, stride=2)
        
        self.bn = nn.BatchNorm2d(nStages[3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def _make_layer(self, num_blocks, in_planes, planes, dropout_rate, stride):
        layers = [WideResNetBlock(in_planes, planes, dropout_rate, stride)]
        for _ in range(1, num_blocks):
            layers.append(WideResNetBlock(planes, planes, dropout_rate, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.relu(self.bn(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18_features(input_size=32):
    """Create ResNet-18 feature extractor. Output dim: 512."""
    model = ResNetFeatures('resnet18', input_size)
    return model


def wrn28_10_features(input_size=32, dropout_rate=0.3):
    """Create WideResNet-28-10 feature extractor. Output dim: 640."""
    model = WideResNetFeatures(depth=28, widen_factor=10, dropout_rate=dropout_rate, input_size=input_size)
    return model


def get_backbone(name, input_size=32, dropout_rate=0.3):
    """
    Factory function to get backbone by name.
    
    Args:
        name: 'vgg16', 'resnet18', or 'wrn28_10'
        input_size: Input image size
        dropout_rate: Dropout rate
    
    Returns:
        (features_module, feature_dim)
    """
    if name == 'vgg16':
        from .vgg_variant import vgg16_variant
        features = vgg16_variant(input_size, dropout_rate)
        return features, 512
    elif name == 'resnet18':
        features = resnet18_features(input_size)
        return features, 512
    elif name == 'wrn28_10':
        features = wrn28_10_features(input_size, dropout_rate)
        return features, 640
    else:
        raise ValueError(f"Unknown backbone: {name}. Choose from: vgg16, resnet18, wrn28_10")
