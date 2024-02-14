import torch.nn as nn

## output_size = (input_size - kernel_size + 2*padding)/stride + 1


class LeNet5(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

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
            nn.Linear(54 * 54 * 120, 84),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(10, num_classes),
        )

    def forward(self, x):
        x = self.convStack(x)
        x = self.fcStack(x)
        return x
