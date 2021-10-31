import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Densenet161(nn.Module):
    def __init__(self):
        super(Densenet161, self).__init__()
        self.densenet = models.densenet161(pretrained=True)
        self.fc1 = nn.Linear(1000, 200)
    def forward(self, x):
        x = self.densenet(x)
        x = self.fc1(x)
        return x