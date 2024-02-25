import lightning as L
import torch
import torch.nn as nn
from torch import optim

from models.full_model import FullModel


class LitDecoder(L.LightningModule):
    def __init__(self, encoder, decoder: nn.Module, lr: float):
        super().__init__()
        encoder.eval()
        self.encoder = encoder

        self.decoder = decoder
        self.lr = lr
        self.loss = torch.nn.functional.mse_loss
        self.test_losses = []

        self.save_hyperparameters(ignore=["decoder"])

    def training_step(self, batch, batch_idx):
        (x, _, label, _) = batch  # x.shape: (200, 1, 64, 64), y.shape: (200, 1, 6)
        with torch.no_grad():
            full_model: FullModel = self.encoder.module
            z = full_model.forward_through_all_cnn_modules(x)
        z = z.detach()
        x_reconstructed = self.decoder(z)
        loss = self.loss(x_reconstructed, x)

        self.log("train_loss", loss)
        return loss

    # validation step
    # def validation_step(self, batch, batch_idx):
    #     (x, _, label, _) = batch
    #     with torch.no_grad():
    #         full_model: FullModel = self.encoder.module
    #         z = full_model.forward_through_all_cnn_modules(x)
    #     z = z.detach()
    #
    #     x_reconstructed = self.decoder(z)
    #     loss = self.loss(x_reconstructed, x, z)
    #     self.log("val_loss", loss)
    #     return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        (x, _, label, _) = batch
        with torch.no_grad():
            full_model: FullModel = self.encoder.module
            z = full_model.forward_through_all_cnn_modules(x)
        z = z.detach()
        x_reconstructed = self.decoder(z)

        loss = self.loss(x_reconstructed, x)
        self.test_losses.append(loss)
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_losses).mean()
        self.log("avg_test_loss", avg_loss)
        self.test_losses = []  # reset for the next epoch
