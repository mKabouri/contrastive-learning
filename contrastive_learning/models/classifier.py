import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, base_model, input_size, hidden_size, n_classes):
        super(MLP, self).__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.training = False
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = self.base_model.one_forward(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
