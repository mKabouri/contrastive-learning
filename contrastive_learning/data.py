import torch
import torchvision.transforms as transforms

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
    Todo more generic class
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

