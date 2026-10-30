"""Deep Learning Architectures for Plant Leaf Disease Classification (CO3 & CO4, Practicals 7 & 8).
Implements:
1. Custom Baseline CNN (3 Conv-BatchNorm-ReLU-Pool blocks + Dense classification head)
2. MobileNetV2 Transfer Learning Architecture (Pre-trained on ImageNet, fine-tuned for 33 classes)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BaselineLeafCNN(nn.Module):
    """Custom Baseline Convolutional Neural Network (Practical 7, CO4).

    Demonstrates understanding of fundamental CNN components:
    - 2D Convolutions with kernel size 3x3
    - Batch Normalization for stabilizing internal covariate shift
    - ReLU non-linear activation
    - Max Pooling (2x2) for spatial downsampling
    - Dropout for regularization
    - Fully Connected (Dense) layers for multi-class classification
    """

    def __init__(self, num_classes=33, in_channels=3):
        super(BaselineLeafCNN, self).__init__()

        # Block 1: 3 -> 32 filters (Input: 224x224 -> Output: 112x112)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 32 -> 64 filters (112x112 -> 56x56)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 64 -> 128 filters (56x56 -> 28x28)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 128 -> 256 filters (28x28 -> 14x14)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # 256 x 4 x 4 = 4096
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.classifier(x)
        return logits


def build_mobilenet_v2(num_classes=33, pretrained=True, freeze_features=False):
    """Build MobileNetV2 with customized classification head (Practical 8, CO4).

    MobileNetV2 uses inverted residual blocks and depthwise separable convolutions,
    making it extremely parameter-efficient (~2.2M params) and fast for edge/laptop execution.
    """
    if pretrained:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2(weights=None)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head: MobileNetV2 features output 1280 channels
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    return model

