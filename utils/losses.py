import torch
import torch.nn as nn

def contrastive_loss(x1, x2, label, margin: float = 1.0):
    dist = nn.functional.pairwise_distance(x1, x2)
    loss = (1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)
    return loss
