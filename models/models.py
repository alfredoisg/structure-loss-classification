import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()

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
            )
            return layer

        self.convStack = nn.Sequential(
            conv_layer(3, 6),
            nn.MaxPool2d(2, 2),  ## 244x244 --> 122x122
            conv_layer(6, 16),
            nn.MaxPool2d(2, 2),  ## 122x122 --> 61x61
            conv_layer(16, 120),
        )

        self.fcStack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(54 * 54 * 120, 84),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(84, 3),
            # nn.Dropout(p=0.5),
            # nn.ReLU(),
            # nn.Linear(10, 3)
        )

    def forward(self, x):
        x = self.convStack(x)
        x = self.fcStack(x)
        return x
