# contrastive-learning

### Datasets:

This project utilizes two datasets:

1. CIFAR-10 Dataset:

- Download from [here](https://www.cs.toronto.edu/~kriz/cifar.html)
- Ensure that the dataset is placed in a folder named cifar10 within the ./data/ directory.


2. Four classes of ImageNet Dataset:

- Download from [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).
- Organize the dataset by creating folders for each class within the ./data/imagenet/ directory. The structure should be similar to ./data/imagenet/classes_folders.

`./data` folder structure:

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

The following are command-line arguments (see ./contrastive_learning/run.py):
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

### Weights:

You can access the weights [here](https://drive.google.com/drive/folders/1j1cKbQuuEvjA8gJJgmhX_B02sR-zkCp_?usp=sharing).

## ./notebooks/

This folder contains a collection of Jupyter notebooks that document our experiments with Contrastive Learning using a downscaled version of the SimCLR framework. Below is an overview of the notebooks and their contents.

## Main Notebook: `Contrastive_Learning.ipynb`

This is the primary notebook that outlines the core implementation of our Contrastive Learning framework. Here, we introduce the foundational concepts, construct the neural network architecture, and detail the training process with the SimCLR approach. It includes the following key sections:

- Dataset loading and preprocessing
- Implementation of data augmentation strategies
- Building the Siamese network with ResNet18 as the encoder
- Training the model with Contrastive Loss
- Visualization of learned representations through t-SNE

## Experiment Notebooks

We have conducted a series of experiments to understand the impact of various hyperparameters and to compare different approaches to representation learning.

### Batch Size Experiments: `Contrastive_Learning_batch_size_experiments.ipynb`

This notebook focuses on the experiments related to batch size, a critical hyperparameter in Contrastive Learning. We have conducted trials with various batch sizes to observe their impact on the quality of the learned representations and the overall performance of the model. Highlights include:

- Analysis of the effect of batch size on model convergence
- Comparison of test accuracies with varying batch sizes
- Discussion on computational constraints like GPU memory limitations

### Comparative Analysis: `Contrastive_Learning_comparative_analysis.ipynb`

In this notebook, we perform a comparative analysis by incorporating different representation learning techniques with our MLP classifier. We explore how different input representations, such as raw images, PCA features, and t-SNE embeddings, influence the classification performance. Key components include:

- Implementation of PCA and t-SNE for dimensionality reduction
- Training the MLP classifier with various input representations
- Comparative evaluation of test accuracies

---

Each notebook is self-contained and includes detailed explanations along with the code, making it easy to follow the experiments and reproduce the results. For the best experience, we recommend starting with the main notebook before proceeding to the experiment notebooks.
