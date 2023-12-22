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

**Note:** All commands should be run from the project root directory.

The following are command-line arguments:
1. --dataset

- Choices: ['cifar10', 'imagenet']
- Default: 'cifar10'
- Description: Choose the dataset for training. Options include 'cifar10' or 'imagenet'.

2. --model

- Choices: ['siamese', 'vanilla', 'transformer']
- Default: 'transformer'
- Description: Choose the model architecture for training. Options include 'siamese', 'vanilla', or 'transformer'.

3. --classifier

- Choices: ['True', 'False']
- Default: 'False'
- Description: Set to 'True' if you want to fine-tune a classifier on a pretrained model; 'False' otherwise.

##### Example of usage:

- To train a siamese model on the CIFAR-10 dataset:
```
python contrastive_learning/run.py --dataset cifar10 --model siamese --classifier False
```

- To fine-tune a classifier on a pretrained transformer model with contrastive_learning:
```
python contrastive_learning/run.py --dataset cifar10 --model transformer --classifier True
```

- To train a vanilla CNN on the ImageNet dataset:
```
python contrastive_learning/run.py --dataset imagenet --model vanilla --classifier False
```

