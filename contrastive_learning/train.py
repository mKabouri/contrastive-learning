import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import data_augmentations as augmentations

def sim(u, v):
    u = u.squeeze(0)
    v = v.squeeze(0)
    return u.dot(v)/(u.norm(2)*v.norm(2))

# def contrastive_loss(rep1, rep2, rep_list, temperature):
#     similarity = sim(rep1, rep2)
#     numerator = torch.exp(similarity/temperature)

#     similarities = torch.stack([sim(rep1, rep)/temperature for rep in rep_list if not torch.equal(rep, rep1)])
#     exp_similiraties = torch.exp(similarities)
#     denominator = torch.sum(exp_similiraties)
#     return -torch.log(numerator/denominator)
def contrastive_loss(rep_tensor, temperature):
    len_reps = len(rep_tensor)

    similarity = torch.tensor([sim(rep_tensor[i], rep_tensor[i+1]) for i in range(0, 2*config.BATCH_SIZE, 2)])
    numerator = torch.exp(similarity/temperature)

    similarities = torch.tensor([sim(rep1, rep)/temperature\
                    for rep1 in rep_tensor\
                    for rep in rep_tensor\
                    if not torch.equal(rep, rep1)]).unfold(0, len_reps, len_reps)

    exp_similiraties = torch.exp(similarities)
    denominator = torch.sum(exp_similiraties)
    loss = -torch.sum(torch.log(numerator/denominator))
    loss.requires_grad = True
    return loss

def train(model, train_data, temperature, optimizer, device=config.device, epochs=config.EPOCHS):
    # We have to specify it because we need it in our model (see model.py)
    model.train()
    model.training = True
    batch_loss = torch.tensor(0., requires_grad=True)
    print(f"Training start with {config.device}")
    for epoch in range(epochs):
        total_loss = 0
        for k, batch in enumerate(train_data):
            batch = batch.to(device)
            rep_list = []
            for image in batch:
                # Our two augmentations
                augmentation1 = augmentations.augment_image
                augmentation2 = augmentations.augment_image

                image_augm1 = augmentation1(image)
                image_augm2 = augmentation2(image)

                output1, output2 = model(image_augm1, image_augm2)
                rep_list.append(output1)
                rep_list.append(output2)

            rep_tensor = torch.stack(rep_list)

            batch_loss = contrastive_loss(rep_tensor, temperature)

            optimizer.zero_grad()

            batch_loss.backward()

            optimizer.step()

            total_loss += batch_loss.item()

        print(f"Epoch: {epoch}, loss: {total_loss/(2*config.BATCH_SIZE)}")
