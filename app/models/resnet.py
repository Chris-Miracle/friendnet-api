from torchvision import models
from torch import nn

def Get_resnet():
  resnet = models.resnet50(pretrained=True)
  num_ftrs = resnet.fc.in_features
  resnet.fc = nn.Linear(num_ftrs, 5)

  return resnet
