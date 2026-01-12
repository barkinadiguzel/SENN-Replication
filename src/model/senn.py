import torch
import torch.nn as nn

class SENN(nn.Module):
    def __init__(self, encoder, concept_encoder, parametrizer, aggregator):
        super().__init__()
        self.encoder = encoder
        self.concept_encoder = concept_encoder
        self.parametrizer = parametrizer
        self.aggregator = aggregator

    def forward(self, x):
        z = self.encoder(x)
        h = self.concept_encoder(z)
        theta = self.parametrizer(z)
        y = self.aggregator(h, theta)

        explanation = {
            "concepts": h,
            "relevances": theta
        }

        return y, explanation
