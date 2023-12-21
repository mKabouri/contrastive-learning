import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import data_augmentations as augmentations
import time
import numpy as np
from tqdm import tqdm

# def sim(u, v):
#     u = u.view(-1)
#     v = v.view(-1)
#     numerator = torch.log(u.dot(v))
#     denominator = torch.log(u.norm(2)) + torch.log(v.norm(2))
#     similarity = torch.exp(numerator - denominator)
#     return similarity

def sim(u, v):
    res = F.cosine_similarity(u, v, dim=-1)
    return res

def contrastive_loss(z1, z2, z_list, temperature):
    similarity_pair = sim(z1, z2)/temperature

    # Calculate the cosine similarity for each pair of z1 and... but not z1 and z1
    similarities = torch.stack([sim(z1, rep)/temperature for rep in z_list if not torch.equal(rep, z1)])

    if similarities.numel() == 0:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)

    # Numerator is exp(similarity_pair)
    exp_similarity = torch.exp(similarity_pair)

    # Denominator is sum(exp(list_similarity_pair))
    exp_similarities = torch.sum(torch.exp(similarities))

    # Apply log-sum-exp trick
    loss = -torch.log(exp_similarity) + torch.log(exp_similarities + exp_similarity)

    if torch.isnan(loss):
        return torch.tensor(0.0, device=z1.device, requires_grad=True)
        
    return loss

def train(model, train_data, temperature, optimizer, device=config.device, epochs=config.EPOCHS, save_weights=True):
    # We have to specify it because we need it in our model (see model.py)
    loss_hist = []
    model.train()
    model.to(device)
    print(f"Training start with {config.device}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Dataset size: {len(train_data)*config.BATCH_SIZE}")
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0

        for i, data in tqdm(enumerate(train_data)):
            images, _ = data.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Initialize total loss for the batch
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Create tuples of transformed images for each image of the batch and put all the zk, zk+1 in a list (feed forward network)
            zk_list = []
            for image in images:
                transformation1 = augmentations.get_transformed_augmented()
                transformation2 = augmentations.get_transformed_augmented()
                transformed_image1 = transformation1(image).to(device)
                transformed_image2 = transformation2(image).to(device)

                z1, z2 = model(transformed_image1.unsqueeze(0), transformed_image2.unsqueeze(0))
                zk_list.append(z1)
                zk_list.append(z2)
                

            # Process each image in the batch
            for i in range(0, len(zk_list), 2):
                # Calculate loss for both the first transformed image and second
                loss1 = contrastive_loss(zk_list[i], zk_list[i+1], zk_list, temperature)
                loss2 = contrastive_loss(zk_list[i+1], zk_list[i], zk_list, temperature)
                batch_loss = batch_loss + (loss1 + loss2)
            
            # Backward pass
            batch_loss.backward()

            # Update weights after processing the entire batch
            optimizer.step()

            total_loss += batch_loss.item()

        average_loss = total_loss/(2*len(images)*len(train_data))
        loss_hist.append(average_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')
        if epoch%5 == 0 and save_weights:
            torch.save(model.state_dict(), config.weights_path + f"/Transformer_weights_{epoch}.pt")

    elapsed_time_seconds = time.time() - start_time
    elapsed_time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))
    print(f'Training finished: In {elapsed_time_formatted}')
    np.save(config.weights_path + "/loss_hist.npy", np.array(loss_hist))


loss_fn = nn.CrossEntropyLoss()

def train_classifier(classifier, train_data, optimizer, device=config.device, epochs=config.EPOCHS, save_weights=True):
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for i, batch in enumerate(train_data):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = classifier(images)

            optimizer.zero_grad()

            batch_loss = loss_fn(outputs, labels)

            batch_loss.backward()

            optimizer.step()
        
            total_loss += batch_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss/len(train_data)*config.BATCH_SIZE
        accuracy = correct_predictions/total_samples

        print(f"Epoch: {epoch}, loss: {avg_loss}, accuracy: {accuracy}")
        if save_weights:
            torch.save(classifier.state_dict(), config.weights_path + f"/Transformer_classifier.pt")

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    print(f"==> Evaluation")
    total_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Compute loss
            total_loss += criterion(output, target).item()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total_samples += target.size(0)

    avg_loss = total_loss/len(test_loader)
    accuracy = 100. * correct/total_samples
    print(f"Evaluation results:\naccuracy: {accuracy}, loss: {avg_loss}")
    return avg_loss, accuracy
