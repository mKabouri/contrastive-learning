import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import config as config
from .utils import ProjectionHead

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18()
        self.head_resnet_in_features = self.resnet.fc.in_features
        self.encoder = nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.proj_head = ProjectionHead(self.head_resnet_in_features)

    def one_forward(self, input):
        output = self.encoder(input)
        output = output.view(output.size(0), -1)
        if not self.training:
            return output
        output = self.proj_head(output)
        return output

    def forward(self, input1, input2):
        output1 = self.one_forward(input1)
        output2 = self.one_forward(input2)
        return output1, output2
