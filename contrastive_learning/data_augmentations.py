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
    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8,\
                                          saturation=0.8, hue=0.2)
    random_grayscale = transforms.RandomGrayscale(0.2)
    color_dist = transforms.Compose([
        transforms.RandomApply([color_jitter], 0.8),
        random_grayscale
    ])
    return color_dist(image)

# def gaussian_blur(image):
#     return transforms.GaussianBlur(config.ORIGINAL_SIZE*0.1)(image)


def augment_image(image):
    return color_distortion(crop_and_resize(image))
