import torch
import torch.nn as nn
import torch.nn.functional as F


def sim(u, v):
    return u.dot(v)/(u.norm(2)*v.norm(2))


def constrative_loss(rep1, rep2, rep_list, tau):
    similarity = sim(rep1, rep2)
    numerator = torch.exp(similarity/tau).item()

    similarities = torch.stack([(sim(rep1, rep)/tau) for rep in rep_list if rep != rep1])
    exp_similiraties = torch.exp(similarities)
    denominator = torch.sum(exp_similiraties)
    return -torch.log(numerator/denominator).item()

def train(model):
    pass