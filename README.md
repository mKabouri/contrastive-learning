# contrastive-learning

- Run files from the root of the project

# ./notebooks/

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
