import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 2048)
        self.fc2 = nn.Linear(2048, 100)

    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = self.fc2(output, use_bias=False)
        return output


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pretrained ResNet-18
        self.resnet = torchvision.models.resnet18()
        # Get the number of input features of the last layer
        self.head_resnet_in_features = self.resnet.fc.in_features
        # Remove the last layer and get the encoder
        self.encoder = nn.Sequential(*(list(self.resnet.children())[:-1]))
        # Add the projection head with the same input features as the output of the encoder
        self.proj_head = ProjectionHead(self.head_resnet_in_features)

    def one_forward(self, x):
        output = self.encoder(x)
        # Flatten the output of the encoder
        output = output.view(output.size(0), -1)
        if not nn.Module.training:
            # Removing the projection head for inference
            return output
        # Forward pass through the projection head
        output = self.proj_head(output)
        return output

    def forward(self, input1, input2):
        output1 = self.one_forward(input1)
        output2 = self.one_forward(input2)
        # Return the output of the projection head for both inputs (both transformed images)
        return output1, output2