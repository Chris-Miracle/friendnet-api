import torch.nn as nn
import torch.nn.functional as F


class FriendNet(nn.Module):
  def __init__(self, num_classes=5):
    super(FriendNet, self).__init__()

    # Convolutional Layers
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2)

    # Batch Norms & Dropouts
    self.bn1 = nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(128)
    self.bn3 = nn.BatchNorm2d(256)

    self.dropout = nn.Dropout(0.5)

    # Dense Layer
    self.fc = nn.Linear(256 * 28 * 28, 512)
    self.out = nn.Linear(512, num_classes)

  def forward(self, x):
    x = self.pool(F.relu(self.bn1(self.conv1(x))))
    x = self.pool(F.relu(self.bn2(self.conv2(x))))
    x = self.pool(F.relu(self.bn3(self.conv3(x))))

    x = x.view(x.size(0), -1)
    x = self.dropout(F.relu(self.fc(x)))
    x = self.out(x)
    return x