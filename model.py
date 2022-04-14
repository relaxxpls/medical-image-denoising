from torch import nn


class PrintShape(nn.Module):
    def __init__(self, key=None):
        super().__init__()

        self.key = key

    def forward(self, x):
        print(self.key, x.shape)

        return x


class AutoEncoderCNN(nn.Module):
    def __init__(self, channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # ? shape of x: (BATCH_SIZE, channels, FEATURE_SHAPE, FEATURE_SHAPE)
        x = self.encoder(x)
        x = self.decoder(x)

        return x
