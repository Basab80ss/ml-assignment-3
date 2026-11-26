import torch
import torch.nn as nn


class MLPPlain(nn.Module):
    """Standard 3-hidden-layer MLP on flattened 32x32x3 CIFAR-10 images (~1.74M params)."""

    def __init__(self, input_dim: int = 3072, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlockMLP(nn.Module):
    """Residual dense block with BatchNorm and ReLU for improved MLP optimization."""

    def __init__(self, width: int):
        super().__init__()
        self.fc = nn.Linear(width, width)
        self.bn = nn.BatchNorm1d(width)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.bn(self.fc(x)))


class MLPResBN(nn.Module):
    """Batch-normalized residual MLP matched to ~1.74M parameters."""

    def __init__(self, input_dim: int = 3072, width: int = 440, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResidualBlockMLP(width)
        self.block2 = ResidualBlockMLP(width)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.head(h)


class MLPMem(nn.Module):
    """High-capacity unregularized MLP (~2.73M params) for the Q1(b) memorisation study."""

    def __init__(self, input_dim: int = 3072, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FairMLP(nn.Module):
    """Parameter-matched MLP (~213k params) with BatchNorm for fair comparison with FairCNN."""

    def __init__(self, input_dim: int = 3072, width: int = 66, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Linear(width, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FairCNN(nn.Module):
    """Compact 4-conv CNN (~214k params) matched to FairMLP for Q2."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x, return_features: bool = False):
        h = self.features(x)
        z = self.proj(h)
        logits = self.classifier(z)
        if return_features:
            return logits, z
        return logits


class ConvAutoencoder(nn.Module):
    """Symmetric convolutional autoencoder with configurable vector bottleneck dimension."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.fc_enc = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.fc_enc(self.encoder_conv(x))

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 128, 4, 4)
        return self.decoder_conv(h)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
