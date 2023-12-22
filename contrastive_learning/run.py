import torch
import torchvision
import torch.optim as optim

import data
import config
import train

from models.siamese_resnet18 import SiameseNetwork
from models.vanilla_cnn import VanillaCNN
from models.vision_transformer import VisionTransformer
from models.classifier import MLP

import data_augmentations as augmentations

import argparse

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train a model on different datasets')
    parser.add_argument('--dataset', choices=['cifar10', 'imagenet'], default='cifar10',
                        help='Choose the dataset (cifar10 or imagenet)')
    parser.add_argument('--model', choices=['siamese', 'vanilla', 'transformer'], default='transformer',
                        help='Choose the model (siamese, vanilla CNN, or transformer)')
    parser.add_argument('--classifier', choices=['True', 'False'], default='False',
                        help='True if do you want to do classification on a pretrained model. False otherwise.')
    return parser.parse_args()

def count_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """
    Main function for training models on different datasets.
    """
    args = parse_arguments()

    if args.dataset == 'cifar10':
        # Training data
        cifar10 = torchvision.datasets.CIFAR10(root=config.cifar10_folder_path, train=True,
                                               download=False, transform=None)
        dataset = data.GetDataset("CIFAR10", cifar10, data.transform_cifar10)
        trainloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=config.BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=2)

        # random_per = torch.randperm(len(cifar10))
        # idx = random_per[:config.NB_SAMPLES].tolist()
        # subset_cifar = dataset[idx]
        # trainloader = torch.utils.data.DataLoader(subset_cifar,
        #                                           batch_size=config.BATCH_SIZE,
        #                                           shuffle=True,
        #                                           num_workers=2)
        
        # Validation data
        cifar10_val = torchvision.datasets.CIFAR10(root=config.cifar10_folder_path, train=False,
                                                  download=False, transform=None)
        dataset_val = data.GetDataset("CIFAR10_Val", cifar10_val, data.transform_cifar10)
        valloader = torch.utils.data.DataLoader(dataset_val,
                                                batch_size=config.BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=2)

        # idx_val = torch.randperm(len(cifar10_val))[:1000].tolist()
        # subset_cifar_val = dataset_val[idx_val]
        # valloader = torch.utils.data.DataLoader(subset_cifar_val,
        #                                         batch_size=config.BATCH_SIZE,
        #                                         shuffle=True,
        #                                         num_workers=2)


    elif args.dataset == 'imagenet':
        training_data = data.SubsetImageNet(config.imagenet_folder_path, data.imagenet_transform)
        trainloader = torch.utils.data.DataLoader(training_data, batch_size=config.BATCH_SIZE, shuffle=True)
    else:
        raise ValueError("Invalid dataset choice. Choose between 'cifar10' and 'imagenet'.")

    if args.model == 'siamese':
        model = SiameseNetwork().to(config.device)
    elif args.model == 'vanilla':
        model = VanillaCNN().to(config.device)
    elif args.model == 'transformer':
        model = VisionTransformer(config.embedding_dim,
                                   config.ORIGINAL_SIZE,
                                   config.attention_dim,
                                   config.nb_heads,
                                   config.nb_layers,
                                   config.patch_size,
                                   config.dropout).to(config.device)
    else:
        raise ValueError("Invalid model choice. Choose between 'siamese', 'vanilla', and 'transformer'.")

    if args.classifier == 'True':
        print(f"Fine-tune classifier with {config.device}")
        model.load_state_dict(torch.load('./weights/Transformer_weights_100.pt'))
        classifier = MLP(model, config.embedding_dim, 512, 10).to(config.device)
        LEARNING_RATE_CLASSIFIER=1e-4
        optimizer = optim.Adam(params=classifier.parameters(), lr=LEARNING_RATE_CLASSIFIER)
        train.train_classifier(classifier, trainloader, optimizer, epochs=11)
        train.evaluate_model(classifier, valloader)
        return
    elif args.classifier == 'False':
        pass
    else:
        raise ValueError("Invalid classifier value. Choose between True or False. (Default: False)")        

    optimizer = optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)
    temperature = 0.5

    print(f"Start training with dataset: {args.dataset}, model: {args.model}")
    print("Number of parameters:", count_parameters(model))
    train.train(model, trainloader, temperature, optimizer, save_weights=True)
    torch.save(model.state_dict(), config.weights_path + f"/{args.model}_weights_final.pt")
    return

if __name__ == "__main__":
    main()
# 6, 6 -> 22 744 064
# 8, 8 -> 29 837 312
# 12, 12 -> 44 023 808
