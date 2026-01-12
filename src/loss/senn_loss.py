import torch
import torch.nn as nn
from torch.autograd import grad

class SENNLoss(nn.Module):
    def __init__(self, lambda_stability=1.0):
        super().__init__()
        self.lambda_stability = lambda_stability
        self.mse = nn.MSELoss()

    def stability_loss(self, x, y, h, theta):
        grad_fx = grad(
            outputs=y.sum(),
            inputs=x,
            create_graph=True
        )[0]

        Jh = []
        for i in range(h.shape[1]):
            grad_hi = grad(
                outputs=h[:, i].sum(),
                inputs=x,
                create_graph=True
            )[0]
            Jh.append(grad_hi)

        Jh = torch.stack(Jh, dim=2)  
        proxy_grad = torch.bmm(Jh, theta.unsqueeze(2)).squeeze(2)

        return self.mse(grad_fx, proxy_grad)

    def forward(self, x, y):
        return self.lambda_stability * self.stability_loss(x, y["y"], y["h"], y["theta"])
