# contrastive-learning

### Datasets:

This project utilizes two datasets:

1. CIFAR-10 Dataset:

- Download from [here](https://www.cs.toronto.edu/~kriz/cifar.html)
- Ensure that the dataset is placed in a folder named cifar10 within the ./data/ directory.


2. Four classes of ImageNet Dataset:

- Download from [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).
- Organize the dataset by creating folders for each class within the ./data/imagenet/ directory. The structure should be similar to ./data/imagenet/classes_folders.

`./data` folder Structure:

```
./
|-- data/
|   |-- cifar10/
|   |   |-- <CIFAR-10 dataset files and folders>
|   |
|   |-- imagenet/
|       |-- classes_folders/
|           |-- <Class 1 images>
|           |-- <Class 2 images>
|           |-- <Class 3 images>
|           |-- <Class 4 images>
```

### ./constrastive-learning

