import torch
import torchvision
import torchvision.transforms as transforms
import random


import config

def crop_and_resize(image):
    return transforms.Compose([
        transforms.RandomResizedCrop(config.ORIGINAL_SIZE, scale=(0.08, 1.0),\
                                     ratio=(0.75, 1.25), antialias=True),
        transforms.RandomHorizontalFlip()
        ])(image)

def color_distortion(image):
    color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1,\
                                          saturation=0.1, hue=0.02)
    random_grayscale = transforms.RandomGrayscale(0.2)
    color_dist = transforms.Compose([
        transforms.RandomApply([color_jitter], 0.8),
        random_grayscale
    ])
    return color_dist(image)

def gaussian_blur(image):
    return transforms.GaussianBlur(config.ORIGINAL_SIZE*0.1)(image)

def augment_image(image):
    return color_distortion(crop_and_resize(image))


def get_transformed_augmented(val):
    if val < 0.5:
        return transforms.Compose([
            transforms.RandomResizedCrop(size=224 ,scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(0.3),
        ])
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                  saturation=0.1, hue=0.05)
    ])

# utils functions
# Data augmentation
def get_transforms(means, stds):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    return transform

# Calculate the mean and std of the subset dataset
def get_mean_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
