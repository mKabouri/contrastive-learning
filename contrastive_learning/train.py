import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import data_augmentations as augmentations
import time
import numpy as np
# def sim(u, v):
#     u = u.view(-1)
#     v = v.view(-1)
#     numerator = torch.log(u.dot(v))
#     denominator = torch.log(u.norm(2)) + torch.log(v.norm(2))
#     similarity = torch.exp(numerator - denominator)
#     return similarity

def sim(u, v):
    # u = u.view(-1)
    # v = v.view(-1)

    res = F.cosine_similarity(u, v, dim=0)
    if np.isnan(res.clone().detach().cpu().numpy()):
        print("IS NAN")
        print(u)
        print(v)
        print("----------------##############")
    return res

def contrastive_loss(rep_tensor, temperature):
    len_reps = len(rep_tensor)

    similarity = torch.tensor([sim(rep_tensor[i], rep_tensor[i+1]) for i in range(0, 2*config.BATCH_SIZE, 2)])

    numerator = torch.exp(similarity/temperature)
    similarities = torch.tensor([sim(rep1, rep)/temperature\
                    for rep1 in rep_tensor\
                    for rep in rep_tensor\
                    if not torch.equal(rep, rep1)]).unfold(0, len_reps, len_reps)
    # print("SIMILARITIES: ", similarities)
    exp_similiraties = torch.exp(similarities)
    # print("EXP_SIM: ", exp_similiraties)
    
    denominator = torch.sum(exp_similiraties)

    loss = -torch.sum(torch.log(numerator) - torch.log(denominator))
    loss.requires_grad = True
    return loss

def train(model, train_data, temperature, optimizer, device=config.device, epochs=config.EPOCHS):
    # We have to specify it because we need it in our model (see model.py)
    model.train()
    model.training = True
    print(f"Training start with {config.device}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Batch size: {config.BATCH_SIZE}")
    for epoch in range(epochs):
        total_loss = 0
        for k, batch in enumerate(train_data):
            batch = batch.to(device)

            augmentation1 = augmentations.augment_image
            augmentation2 = augmentations.augment_image
            # print("We augment")
            # start_aug = time.time()
            batch_augm1 = augmentation1(batch)
            batch_augm2 = augmentation2(batch)
            # end_aug = time.time() - start_aug
            # print("Finish augment", end_aug)

            # print("Start_inference")
            # start_inf = time.time()
            output1, output2 = model(batch_augm1, batch_augm2)
            # end_inf = time.time() - start_inf
            # print("Finish inf", end_inf)

            rep_tensor = torch.stack((output1, output2)).view(-1, config.REP_OUTPUT)

            batch_loss = contrastive_loss(rep_tensor, temperature)

            optimizer.zero_grad()

            batch_loss.backward()

            optimizer.step()

            total_loss += batch_loss.item()

        print(f"Epoch: {epoch}, loss: {total_loss/3900}")
