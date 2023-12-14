import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

ORIGINAL_SIZE=224
BATCH_SIZE=26
LEARNING_RATE=1e-3
EPOCHS=60
REP_OUTPUT=512 # Output of the projection head

###############################
##### Transformer configs #####
###############################
embedding_dim = 3072
attention_dim = 12
patch_size = 32
nb_heads = 12
nb_layers = 12
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

################################
######### Weights path #########
################################
weights_path = os.path.join(os.path.curdir, "weights")
