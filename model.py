import torch.nn as nn
import timm

class Vit_large_patch16_224(nn.Module):
    def __init__(self):
        super(Vit_large_patch16_224, self).__init__()
        self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
        self.fc1 = nn.Linear(1000, 200)
    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        return x

