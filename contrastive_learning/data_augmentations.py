import torchvision
import torchvision.transforms as transforms

import config

def crop_and_resize(image):
    return transforms.Compose([
        transforms.RandomResizedCrop(config.ORIGINAL_SIZE, scale=(0.3, 0.4),\
                                     ratio=(0.8, 1.2), antialias=True),
        transforms.RandomHorizontalFlip()
        ])(image)

def color_distortion(image):
    return transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 1),\
                                  saturation=(0.1, 2), hue=0.2)(image)

def gaussian_blur(image):
    return transforms.GaussianBlur(3)(image)

def augment_image(image):
    return gaussian_blur(color_distortion(crop_and_resize(image)))
