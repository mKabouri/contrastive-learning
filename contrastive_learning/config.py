import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

ORIGINAL_SIZE=32
LEARNING_RATE=1
BATCH_SIZE=10
EPOCHS=100
REP_OUTPUT=100


NB_SAMPLES=100

# List all dataset that we will work on
cifar10_folder_path = os.path.join(os.path.curdir, "data/cifar10")
