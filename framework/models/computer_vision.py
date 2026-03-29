from torch import nn


class SplitLeNet5V1(nn.Module):
    """
    LeNet5V1 split into embed (ϕ) + head (h) for HCFL and LCFed.
      - embed : feature + classifier[:-1]  → 84-dim embedding
      - head  : classifier[-1]             → Linear(84 → 10)
    """
    def __init__(self):
        super().__init__()
        base = LeNet5V1()
        self.embed = nn.Sequential(
            base.feature,
            *list(base.classifier.children())[:-1]
        )
        self.head = list(base.classifier.children())[-1]

    def forward(self, x):
        return self.head(self.embed(x))


class LeNet5V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # 28*28->32*32-->28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14

            # 2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        return self.classifier(self.feature(x))


class CNNCifar(nn.Module):
    """
    Simple CNN for CIFAR-10 / CIFAR-100 (32x32 RGB input).
    num_classes=10 for CIFAR-10, 100 for CIFAR-100.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.feature(x))


class SplitCNNCifar(nn.Module):
    """
    Split version of CNNCifar for HCFL and LCFed.
      - embed : feature + FC(4096 → 512) → 512-dim embedding
      - head  : Linear(512 → num_classes)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
        )
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.head(self.embed(x))


class LeNet5FEMNIST(nn.Module):
    """LeNet5 for FEMNIST (28×28 grayscale, 62 classes by default)."""
    def __init__(self, num_classes=62):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.feature(x))


class SplitLeNet5FEMNIST(nn.Module):
    """
    Split LeNet5 for FEMNIST (HCFL/LCFed/FedPer).
      - embed : feature + classifier[:-1] → 84-dim embedding
      - head  : Linear(84 → 62)
    """
    def __init__(self, num_classes=62):
        super().__init__()
        base = LeNet5FEMNIST(num_classes)
        self.embed = nn.Sequential(
            base.feature,
            *list(base.classifier.children())[:-1]
        )
        self.head = list(base.classifier.children())[-1]

    def forward(self, x):
        return self.head(self.embed(x))


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)