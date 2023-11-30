from torchvision import transforms


def get_transforms():
    # Return a list of transformation, needs to be modified
    return transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.3, 0.4), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip()
    ])


def color_distortion():
    return transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 1),
                                  saturation=(0.1, 2), hue=0.2)


def gaussian_blur():
    return transforms.GaussianBlur(3)

