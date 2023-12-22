import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

ORIGINAL_SIZE=224
BATCH_SIZE=100
LEARNING_RATE=1e-4
EPOCHS=35
REP_OUTPUT=512 # Output of the projection head

###############################
##### Transformer configs #####
###############################
embedding_dim = 768
attention_dim = embedding_dim
patch_size = 16
nb_heads = 6
nb_layers = 6
dropout = 0.2
###############################
####### CIFAR10 configs #######
###############################
NB_SAMPLES=10000

# List all dataset that we will work on
cifar10_folder_path = os.path.join(os.path.curdir, "data/cifar10")

###############################
###### IMAGE_NET configs ######
###############################
imagenet_folder_path = os.path.join(os.path.curdir, "data/imagenet")

################################
######### Weights path #########
################################
weights_path = os.path.join(os.path.curdir, "weights")
