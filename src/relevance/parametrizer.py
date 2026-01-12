import torch
import torch.nn as nn

class Parametrizer(nn.Module):
    def __init__(self, hidden_dim, concept_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, concept_dim)
        )

    def forward(self, z):
        return self.net(z)
