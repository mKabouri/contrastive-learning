import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import config

def transform_loaded_data(means, stds):
    return transforms.Compose([
        transforms.Resize(config.ORIGINAL_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

transform_cifar10 = transforms.Compose([
        transforms.Resize(config.ORIGINAL_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])

class GetDataset(torch.utils.data.Dataset):
    """
    Generic dataset class for handling various datasets with specified transformations.

    Attributes:
        name (str): Name of the dataset.
        dataset (torch.utils.data.Dataset): Original dataset.
        transform (torchvision.transforms.Compose): Transformation to be applied to the dataset.
    """
    def __init__(self, name, dataset, transform=None):
        self.name = name
        self.dataset = dataset
        self.transform = transform

    def __name__(self):
        return self.name
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(index, int): # This is our previous dataset
            image, label = self.dataset[index]
            if self.transform:
                image = self.transform(image)
            return image, label
        elif isinstance(index, list):
            items = [(self.dataset[i][0], self.dataset[i][1]) for i in index]
            if self.transform:
                items = [(self.transform(img), label) for img, label in items]
            return items
        else:
            raise TypeError("Index must be an integer or a list of integers")

imagenet_transform = transforms.Compose([
        transforms.Resize((config.ORIGINAL_SIZE, config.ORIGINAL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

class SubsetImageNet(torch.utils.data.Dataset):
    """
    Dataset class for a subset of ImageNet, with specified transformations.

    Attributes:
        root_folder (str): Root folder containing the subset of ImageNet.
        transform (torchvision.transforms.Compose): Transformation to be applied to the dataset.
        classes (list): List of class names in the subset.
        class_to_idx (dict): Mapping from class names to indices.
        idx_to_class (dict): Mapping from indices to class names.
        img_paths (list): List of image paths and corresponding labels.
    """
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
            self.img_paths.extend([(os.path.join(class_path, img), self.class_to_idx[cls]) for img in img_files])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label