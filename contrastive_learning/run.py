import torch
import torchvision
import torch.optim as optim

import data
import config
import train
import model
import load_imagenet as imagenet

####################################################
##################### DATASETS #####################
####################################################

# CIFAR 10 Dataset
# cifar10 = torchvision.datasets.CIFAR10(root=config.cifar10_folder_path, train=True,\
#                                            download=False, transform=None)
# dataset_cifar = data.UnlabeledDataset("CIFAR10", cifar10, data.transform)
# trainloader_cifar10 = torch.utils.data.DataLoader(dataset_cifar,\
#                                                   batch_size=config.BATCH_SIZE,\
#                                                   shuffle=True, num_workers=2)

# # Train on small dataset
# random_per = torch.randperm(len(cifar10))
# idx = random_per[:config.NB_SAMPLES].tolist()
# subset_cifar = dataset_cifar[idx]
# trainloader_subsetcifar = torch.utils.data.DataLoader(subset_cifar,\
#                                                       batch_size=config.BATCH_SIZE,\
#                                                       shuffle=True, num_workers=2)


#####################
##### IMAGE_NET #####
#####################
cups_dataset = imagenet.LoadSubImageNet(config.classe_paths["cup"], imagenet.imagenet_transform)
birds_dataset = imagenet.LoadSubImageNet(config.classe_paths["cup"], imagenet.imagenet_transform)
car_dataset = imagenet.LoadSubImageNet(config.classe_paths["cup"], imagenet.imagenet_transform)



training_data = torch.utils.data.ConcatDataset([cups_dataset, birds_dataset, car_dataset])
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=config.BATCH_SIZE, shuffle=True)

print(len(training_data))
print(training_data[0][0].size())
print(next(iter(training_dataloader))[0][0].size())

####################################################
####################################################
####################################################

simclr_model = model.Siamese_Network().to(config.device)
# simclr_model = model.VanillaCNN().to(config.device)

# We will use Adam optimizer for now
optimizer = optim.Adam(params=simclr_model.parameters(), lr=config.LEARNING_RATE)

temperature = 0.5

if __name__ == "__main__":
    print("Start")
    train.train(simclr_model, training_dataloader, temperature, optimizer)

    torch.save(simclr_model.state_dict(), "Siamese_Net_weights.pt")
