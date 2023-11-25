import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Projection_Head(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 2048)
        self.fc2 = nn.Linear(2048, 100)
    
    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = self.fc2(output)
        return output

class Siamese_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()
        self.head_resnet_in_features = self.resnet.fc.in_features
        self.encoder = nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.proj_head = Projection_Head(self.head_resnet_in_features)

    def one_forward(self, input):
        output = self.encoder(input)
        output = output.view(output.size(0), -1)
        if not nn.Module.training:
            return output
        output = self.proj_head(output)
        return output

    def forward(self, input1, input2):
        output1 = self.one_forward(input1)
        output2 = self.one_forward(input2)
        return output1, output2 