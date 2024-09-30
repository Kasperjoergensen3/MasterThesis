# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from torch.optim import Adam

import monai
from typing import Optional

# from monai-generative
from generative.losses.perceptual import PerceptualLoss
from generative.losses.adversarial_loss import PatchAdversarialLoss

from model.losses.kl_loss import KLLoss3D


class Lightning3DVaeGanPerceptual(pl.LightningModule):

    def __init__(
        self,
        g_network: nn.Module,
        g_optimizer: torch.optim.Optimizer,
        d_network: nn.Module,
        d_optimizer: torch.optim.Optimizer,
        recon_loss: Optional[nn.Module] = None,
        adv_loss: Optional[nn.Module] = None,
        p_loss: Optional[nn.Module] = None,
        kl_loss: Optional[nn.Module] = None,
        d_train_steps: int = 1,
        recon_weight: float = 1.0,
        adv_weight: float = 0.1,
        p_weight: float = 0.1,
        kl_weight: float = 1e-7,
    ) -> None:
        super().__init__()
        self.g_network = g_network
        self.g_optimizer = g_optimizer
        self.d_network = d_network
        self.d_optimizer = d_optimizer
        self.d_train_steps = d_train_steps

        # some fixed hyperparameters not controllable by the user from the config
        self.spatial_dims = 3
        # loss weights for generator
        # kl_weight: important hyper-parameter.
        #     If too large, decoder cannot recon good results from latent space.
        #     If too small, latent space will not be regularized enough for the diffusion model
        self.recon_weight = recon_weight
        self.adv_weight = adv_weight
        self.p_weight = p_weight
        self.kl_weight = kl_weight

        # losses
        self.setup_losses(recon_loss, adv_loss, p_loss, kl_loss)

        # params for the optimizer
        self.g_params = self.g_network.parameters()
        self.g_optimizer = Adam
        self.g_lr = 1e-05
        self.d_optimizer = Adam
        self.d_params = self.d_network.parameters()
        self.d_lr = 1e-05

    def setup_losses(self, recon_loss, adv_loss, p_loss, kl_loss):
        # set the losses and intialize default losses if not provided
        self.recon_loss = recon_loss if recon_loss is not None else nn.L1Loss()
        self.adv_loss = (
            adv_loss
            if adv_loss is not None
            else PatchAdversarialLoss(criterion="least_squares")
        )
        self.p_loss = (
            p_loss
            if p_loss is not None
            else PerceptualLoss(
                self.spatial_dims,
                network_type="resnet50",
                is_fake_3d=True,
                fake_3d_ratio=0.2,
                pretrained=False,
            )
        )
        self.kl_loss = kl_loss if kl_loss is not None else KLLoss3D()

    def g_loss(self, x_hat, x, z_mu, z_sigma, return_all_losses=False):
        recon_loss = self.recon_loss(x_hat, x) if self.recon_weight > 0 else 0
        kl_loss = self.compute_kl_loss(z_mu, z_sigma) if self.kl_weight > 0 else 0
        p_loss = self.p_loss(x_hat.float(), x.float()) if self.p_weight > 0 else 0
        if self.adv_weight > 0:
            logits_fake = self.d_network(x_hat)[-1]
            adv_loss = self.adv_loss(
                logits_fake, target_is_real=True, for_discriminator=False
            )
        else:
            adv_loss = 0
        g_loss = (
            self.recon_weight * recon_loss
            + self.kl_weight * kl_loss
            + self.p_weight * p_loss
            + self.adv_weight * adv_loss
        )
        if return_all_losses:
            return g_loss, recon_loss, kl_loss, p_loss, adv_loss
        return g_loss

    def discriminator_loss(self, gen_images, real_images):
        logits_fake = self.d_network(gen_images.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(
            logits_fake, target_is_real=False, for_discriminator=True
        )
        logits_real = self.d_network(real_images.contiguous().detach())[-1]
        loss_d_real = self.adv_loss(
            logits_real, target_is_real=True, for_discriminator=True
        )
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
        loss_d = self.adv_weight * discriminator_loss
        return loss_d

    def prepare_batch(self, batch):
        return batch  # no need to change anything at the moment

    def training_step(self, batch, batch_idx, optimizer_idx):
        # training_step defined the train loop. It is independent of forward
        # instead of x = batch. there is no y as it is an autoencoder
        x = self.prepare_batch(batch)
        # forward pass through the generator network to get the reconstructed image
        x_hat, z_mu, z_sigma = self.g_network(x)

        #  Train Generator through the GAN
        if optimizer_idx == 0:
            # freeze discriminator
            for param in self.d_network.parameters():
                param.requires_grad = False
            # calculate the generator loss
            g_loss, recon_loss, kl_loss, p_loss, adv_loss = self.g_loss(
                x_hat, x, z_mu, z_sigma, return_all_losses=True
            )
            # log all different losses
            self.log("g_loss_train", g_loss)
            self.log("g_recon_loss_train", recon_loss)
            self.log("g_kl_loss_train", kl_loss)
            self.log("g_p_loss_train", p_loss)
            self.log("g_adv_loss_train", adv_loss)
            return g_loss

        #  Train Discriminator
        if optimizer_idx == 1:
            # unfreeze discriminator
            for param in self.d_network.parameters():
                param.requires_grad = True
            # Consider implemeting loop for multiple discriminator training steps
            # ...
            # return the discriminator loss
            d_loss = self.discriminator_loss(x_hat, x)
            # log all losses
            self.log("d_loss_train", d_loss)
            return d_loss

    def validation_step(self, val_batch, batch_idx):
        x = self.prepare_batch(val_batch)  # instead of x, y = val_batch
        x_hat, z_mu, z_sigma = self.g_network(x)
        # log all different losses
        g_loss, recon_loss, kl_loss, p_loss, adv_loss = self.g_loss(
            x_hat, x, z_mu, z_sigma, return_all_losses=True
        )
        d_loss = self.discriminator_loss(x_hat, x)

        # log all losses
        self.log("g_loss_val", g_loss)
        self.log("g_recon_loss_val", recon_loss)
        self.log("g_kl_loss_val", kl_loss)
        self.log("g_p_loss_val", p_loss)
        self.log("g_adv_loss_val", adv_loss)
        self.log("d_loss_val", d_loss)
        return g_loss

    def configure_optimizers(self):
        # can return multiple optimizers in a tuple, for GANs for instance
        d_optimizer = self.g_optimizer
        g_optimizer = self.d_optimizer
        return [g_optimizer, d_optimizer]

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.g_network.reconstruct(x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # returns the reconstructed image, the mean and the standard deviation of the latent space
        return self.g_network(x)
