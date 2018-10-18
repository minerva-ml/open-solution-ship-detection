import torch.nn as nn
import pretrainedmodels

class Densenet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.features = pretrainedmodels.__dict__['densenet201'](num_classes=1000, pretrained=pretrained)
        self.classifier = nn.Linear(in_features=1000, out_features=2)

    def forward(self, input):
        x = self.features(input)
        x = self.classifier(x)
        return x
