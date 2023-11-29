import torch
import torchvision
import torch.optim as optim

import data
import config
import train
import model

cifar10 = torchvision.datasets.CIFAR10(root=config.cifar10_folder_path, train=True,\
                                           download=True, transform=None)
dataset_cifar = data.UnlabeledDataset("CIFAR10", cifar10, data.transform)
trainloader_cifar10 = torch.utils.data.DataLoader(dataset_cifar,\
                                                  batch_size=config.BATCH_SIZE,\
                                                  shuffle=True, num_workers=2)

simclr_model = model.Siamese_Network().to(config.device)

# We will use Adam optimizer for now
optimizer = optim.Adam(params=simclr_model.parameters(), lr=config.LEARNING_RATE)

temperature = 0.5

if __name__ == "__main__":
    train.train(simclr_model, trainloader_cifar10, temperature, optimizer)