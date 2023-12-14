import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

ORIGINAL_SIZE=224
BATCH_SIZE=10
LEARNING_RATE=0.3*BATCH_SIZE/256
EPOCHS=100
REP_OUTPUT=512 # Output of the projection head

###############################
##### Transformer configs #####
###############################
embedding_dim = 768
attention_dim = 12
nb_heads = 6
nb_layers = 6
patch_size = 16
dropout = 0.2

###############################
####### CIFAR10 configs #######
###############################
NB_SAMPLES=20

# List all dataset that we will work on
cifar10_folder_path = os.path.join(os.path.curdir, "data/cifar10")

###############################
###### IMAGE_NET configs ######
###############################
imagenet_folder_path = os.path.join(os.path.curdir, "data/imagenet")

classe_paths = {
    "car": imagenet_folder_path + "/car",
    "cup": imagenet_folder_path + "/cup",
    "bird": imagenet_folder_path + "/bird",
    "fish": imagenet_folder_path + "/fish",
    "wash_machine": imagenet_folder_path + "/wash_machine"
}

labels_dict = {
    classe_paths["car"]: 0,
    classe_paths["cup"]: 1,
    classe_paths["bird"]: 2,
    classe_paths["fish"]: 3,
    classe_paths["wash_machine"]: 4
}
