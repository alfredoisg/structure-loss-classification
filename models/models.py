import torch.nn as nn
from torchvision import models

## output_size = (input_size - kernel_size + 2*padding)/stride + 1


class LeNet5(nn.Module):
    def __init__(self, num_classes: int, size_layer_1: int, size_layer_2: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.size_layer_1 = size_layer_1
        self.size_layer_2 = size_layer_2

        def conv_layer(in_channels, out_channels):
            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=5,
                    stride=1,
                    padding=0,
                ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
            )
            return layer

        self.convStack = nn.Sequential(
            conv_layer(3, 6),  ## 244x244 --> (244 - 5 + 2*0)/1 +1 = 240x240
            nn.MaxPool2d(2, 2),  ## 240/2 --> 120x120
            conv_layer(6, 16),  ## 120x120 --> (120 -5 + 2*0)/1 +1 = 116x116
            nn.MaxPool2d(2, 2),  ## 116/2 --> 58x58
            conv_layer(16, 120),  ## 58x58 --> (58 -5 + 2*0)/1 +1 = 54x54
        )

        self.fcStack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(54 * 54 * 120, self.size_layer_1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(self.size_layer_1, self.size_layer_2),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(self.size_layer_2, num_classes),
        )

    def forward(self, x):
        x = self.convStack(x)
        x = self.fcStack(x)
        return x


class VGG16(nn.Module):
    def __init__(
        self, num_classes: int, size_layer_1: int, size_layer_2: int, size_layer_3: int
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.size_layer_1 = size_layer_1
        self.size_layer_2 = size_layer_2
        self.size_layer_3 = size_layer_3

        def conv_layer(in_channels, out_channels):

            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,  # The padding value of 1 ensures  that the spatial dimensions of the feature maps do not decrease after the convolution operation
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

            return layer

        def fc_layer(in_features, out_features):

            layer = nn.Linear(in_features=in_features, out_features=out_features)

            nn.init.kaiming_normal_(
                layer.weight, nonlinearity="relu"
            )  # weight initialization method
            ## The purpose of weight initialization is to set the initial values of the weights in the model so that the training process can start from a good starting point.
            return layer

        self.convStack = nn.Sequential(
            # C64-C64-P2
            conv_layer(3, 64),
            conv_layer(64, 64),
            nn.MaxPool2d(2, 2),  # 244x244 --> 122x122
            # C128-C128-P2
            conv_layer(64, 128),
            conv_layer(128, 128),
            nn.MaxPool2d(2, 2),  # 122x122 --> 61x61
            # C256-C256-C256-P2
            conv_layer(128, 256),
            conv_layer(256, 256),
            conv_layer(256, 256),
            conv_layer(256, 256),
            nn.MaxPool2d(2, 2),  # 61x61 --> 30x30
            # C512-C512-C512-P2
            conv_layer(256, 512),
            conv_layer(512, 512),
            conv_layer(512, 512),
            nn.MaxPool2d(2, 2),  # 30x30 --> 15x15
            # C512-C512-C512-P2
            conv_layer(512, 512),
            conv_layer(512, 512),
            conv_layer(512, 512),
            nn.MaxPool2d(2, 2),  # 15x15 --> 7x7
        )

        self.fullyConStack = nn.Sequential(
            nn.Flatten(),
            # F4096-F4096-F1000
            fc_layer(512 * 7 * 7, self.size_layer_1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            fc_layer(self.size_layer_1, self.size_layer_2),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            fc_layer(self.size_layer_2, self.size_layer_3),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            fc_layer(self.size_layer_3, self.num_classes),
        )

    def forward(self, x):

        x = self.convStack(x)
        x = self.fullyConStack(x)

        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes) -> None:
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.resnet18.fc = nn.Linear(
            in_features=self.resnet18.fc.in_features, out_features=self.num_classes
        )

    def forward(self, x):
        out = self.resnet18(x)
        return out
