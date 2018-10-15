import torch.nn as nn
from torchvision.models import resnet101


class Resnet101(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.model = resnet101(pretrained=pretrained)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=1)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        x = self.model(input)
        return x.squeeze()