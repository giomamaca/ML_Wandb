import torch
import torch.nn as nn

class MLP(nn.Module):
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

class _ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.short(x))

class ResNetMini(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(_ResBlock(64, 64), _ResBlock(64, 64))
        self.layer2 = nn.Sequential(_ResBlock(64, 128, 2), _ResBlock(128, 128))
        self.layer3 = nn.Sequential(_ResBlock(128, 256, 2), _ResBlock(256, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.head(x)

class AlexNetSmall(nn.Module):
    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

class _Inception(nn.Module):
    def __init__(self, cin, c1, c3r, c3, c5r, c5, cpool):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(cin, c1, 1), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(
            nn.Conv2d(cin, c3r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c3r, c3, 3, padding=1), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(
            nn.Conv2d(cin, c5r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c5r, c5, 5, padding=2), nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1), nn.Conv2d(cin, cpool, 1), nn.ReLU(inplace=True))
    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class InceptionSmall(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.inc1 = _Inception(96, 64, 96, 128, 16, 32, 32)
        self.inc2 = _Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool = nn.MaxPool2d(2)                             # 12->6
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(480, num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.inc1(x); x = self.inc2(x); x = self.pool(x)
        return self.head(x)


