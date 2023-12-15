import torch
import torchvision
import torch.optim as optim

import data
import config
import train

from models.siamese_resnet18 import SiameseNetwork
from models.vanilla_cnn import VanillaCNN
from models.vision_transformer import VisionTransformer
import load_imagenet as imagenet
import data_augmentations as augmentations

####################################################
##################### DATASETS #####################
####################################################

# CIFAR 10 Dataset
# cifar10 = torchvision.datasets.CIFAR10(root=config.cifar10_folder_path, train=True,\
#                                            download=False, transform=None)
# dataset_cifar = data.UnlabeledDataset("CIFAR10", cifar10, data.transform_cifar10)
# trainloader_cifar10 = torch.utils.data.DataLoader(dataset_cifar,\
#                                                   batch_size=config.BATCH_SIZE,\
#                                                   shuffle=True, num_workers=2)

# # Train on small dataset
# random_per = torch.randperm(len(cifar10))
# idx = random_per[:config.NB_SAMPLES].tolist()
# subset_cifar = dataset_cifar[idx]
# training_dataloader = torch.utils.data.DataLoader(subset_cifar,\
#                                                       batch_size=config.BATCH_SIZE,\
#                                                       shuffle=True, num_workers=2)


#####################
##### IMAGE_NET #####
#####################
training_data = imagenet.SubsetImageNet(config.imagenet_folder_path, imagenet.imagenet_transform)
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=config.BATCH_SIZE, shuffle=True)

####################################################
####################################################
####################################################

# simclr_model = SiameseNetwork().to(config.device)
# simclr_model = VanillaCNN().to(config.device)
simclr_model = VisionTransformer(config.embedding_dim,
                                 config.ORIGINAL_SIZE,
                                 config.attention_dim,
                                 config.nb_heads,
                                 config.nb_layers,
                                 config.patch_size,
                                 config.dropout).to(config.device)

# We will use Adam optimizer for now
optimizer = optim.Adam(params=simclr_model.parameters(), lr=config.LEARNING_RATE)
# print(len(training_data))
temperature = 0.5

if __name__ == "__main__":
    print("Start")
    train.train(simclr_model, training_dataloader, temperature, optimizer,save_weights=True)

    torch.save(simclr_model.state_dict(), config.weights_path + "/Transformer_weights_final.pt")
