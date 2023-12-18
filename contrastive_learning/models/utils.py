import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

import config as config

class ProjectionHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, config.REP_OUTPUT)
    
    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = self.fc2(output)
        return output

def image_to_patches(x, patch_size, flatten_channels=True):
    """
    Adapted from here:
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
    """
    if x.dim() == 3:
        C, H, W = x.shape
        x = x.reshape(C, H//patch_size, patch_size, W//patch_size, patch_size)
        x = x.permute(1, 3, 0, 2, 4)
        x = x.flatten(0,1)
        if flatten_channels:
            x = x.flatten(1,3) 
        return x
    elif x.dim() == 4: # For batches
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(1,2)
        if flatten_channels:
            x = x.flatten(2,4) 
        return x
    else:
        raise ValueError("Input must have 3 or 4 dimensions.")


if __name__ == '__main__':
    image = Image.open("./contrastive_learning/models/test_image.JPEG")
    tensor_image = torchvision.transforms.functional.pil_to_tensor(image)
    tensor_image = torchvision.transforms.Resize((228, 228))(tensor_image)
    PATCH_SIZE=76
    patches = image_to_patches(tensor_image, PATCH_SIZE)
    print(patches.size())   
    plt.imshow(tensor_image.permute(1, 2, 0), vmin=0, vmax=255)
    plt.show()
    plt.imshow(patches[0].permute(1, 2, 0), vmin=0, vmax=255)
    plt.show()
    plt.imshow(patches[1].permute(1, 2, 0), vmin=0, vmax=255)
    plt.show()
    plt.imshow(patches[2].permute(1, 2, 0), vmin=0, vmax=255)
    plt.show()
    plt.imshow(patches[3].permute(1, 2, 0), vmin=0, vmax=255)
    plt.show()
    print(tensor_image)
    print(patches[1])