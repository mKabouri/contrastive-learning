import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import config

imagenet_transform = transforms.Compose([
        transforms.Resize((config.ORIGINAL_SIZE, config.ORIGINAL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

class LoadSubImageNet(torch.utils.data.Dataset):
    def __init__(self, classe_dir, transform=None):
        super(LoadSubImageNet, self).__init__()
        self.classe_dir = classe_dir
        self.img_labels = np.array([config.labels_dict[self.classe_dir] for _ in range(len(os.listdir(self.classe_dir)))])
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if isinstance(idx, int): # This is our previous dataset
            img_path = os.path.join(self.classe_dir, os.listdir(self.classe_dir)[idx])
            image = Image.open(img_path)
            label = self.img_labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        elif isinstance(idx, list):
            img_paths = [os.path.join(self.classe_dir,\
                                      os.listdir(self.classe_dir)[index])\
                                        for index in idx]
            images = [Image.open(img_path) for img_path in img_paths]
            labels = [self.img_labels[index] for index in idx]
            if self.transform:
                images = [self.transform(img) for img in images]
            return images, labels
        else:
            raise TypeError("Index must be an integer or a list of integers")

