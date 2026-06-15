import torch.nn as nn

class MLP(nn.Module):
    """Iteration 1: baseline - expected to underfit (flatten loses spatial structure)"""
    def __init__(self, hidden=512, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48*48, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

class SmallCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128*6*6, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

class RegularizedCNN(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout(dropout)
            )
        self.features = nn.Sequential(block(1, 64), block(64, 128), block(128, 256))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*6*6, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))
