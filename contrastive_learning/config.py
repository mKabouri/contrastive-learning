import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

ORIGINAL_SIZE=224
LEARNING_RATE=1e-3
BATCH_SIZE=4
EPOCHS=1000

# List all dataset that we will work on
cifar10_folder_path = os.path.join(os.path.curdir, "data/cifar10")
