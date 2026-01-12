import torch
import torch.nn as nn

class Aggregator(nn.Module):
    def forward(self, h, theta):
        z = h * theta
        y = z.sum(dim=1, keepdim=True)
        return y
