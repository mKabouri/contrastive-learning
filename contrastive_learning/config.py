import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

ORIGINAL_SIZE=100
LEARNING_RATE=4
BATCH_SIZE=60
EPOCHS=10

# List all dataset that we will work on
cifar10_folder_path = os.path.join(os.path.curdir, "data/cifar10")
