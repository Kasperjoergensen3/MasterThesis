import torch
import torch.nn as nn


class KLLoss3D(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, z_mu, z_sigma):
        # Calculate the KL loss
        kl_loss = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
            dim=[1, 2, 3, 4],
        )
        # Return the mean KL loss over the batch
        if self.reduction == "mean":
            return torch.mean(kl_loss)
        elif self.reduction == "sum":
            return torch.sum(kl_loss)
        else:
            return kl_loss
