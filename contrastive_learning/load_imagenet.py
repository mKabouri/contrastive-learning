import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import config

imagenet_transform = transforms.Compose([
        transforms.Resize((config.ORIGINAL_SIZE, config.ORIGINAL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

class SubsetImageNet(torch.utils.data.Dataset):
    def __init__(self, root_folder, transform=None):
        super(SubsetImageNet, self).__init__()

        self.root_folder = root_folder
        self.classes = os.listdir(root_folder)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.transform = transform

        self.img_paths = []

        for cls in self.classes:
            class_path = os.path.join(root_folder, cls)
            img_files = os.listdir(class_path)
            self.img_paths.extend([os.path.join(class_path, img) for img in img_files])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
