import torch
import torchvision
import torch.optim as optim

import data
import config
import train
import model

####################################################
##################### DATASETS #####################
####################################################

# CIFAR 10 Dataset
cifar10 = torchvision.datasets.CIFAR10(root=config.cifar10_folder_path, train=True,\
                                           download=True, transform=None)
dataset_cifar = data.UnlabeledDataset("CIFAR10", cifar10, data.transform)
trainloader_cifar10 = torch.utils.data.DataLoader(dataset_cifar,\
                                                  batch_size=config.BATCH_SIZE,\
                                                  shuffle=True, num_workers=2)

# Train on small dataset
random_per = torch.randperm(len(cifar10))
idx = random_per[:3000].tolist()
subset_cifar = dataset_cifar[idx]
trainloader_subsetcifar = torch.utils.data.DataLoader(subset_cifar,\
                                                      batch_size=config.BATCH_SIZE,\
                                                      shuffle=True, num_workers=2)

####################################################
####################################################
####################################################

simclr_model = model.Siamese_Network().to(config.device)
# simclr_model = model.VanillaCNN().to(config.device)

# We will use Adam optimizer for now
optimizer = optim.Adam(params=simclr_model.parameters(), lr=config.LEARNING_RATE)

temperature = 0.5

if __name__ == "__main__":
    train.train(simclr_model, trainloader_subsetcifar, temperature, optimizer)

    torch.save(simclr_model.state_dict(), "Siamese_Net_weights.pt")
