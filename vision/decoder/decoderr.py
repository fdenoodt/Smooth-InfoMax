import lightning as L
import torch
import torch.nn as nn
from torch import optim

from config_code.config_classes import DecoderLoss
from decoder.decoder_losses import MSE_Loss
from vision.models.FullModel import FullVisionModel


class Decoder(L.LightningModule):
    def __init__(self, encoder: FullVisionModel, lr: float, loss: DecoderLoss):
        super().__init__()
        encoder.eval()
        self.encoder: FullVisionModel = encoder
        self.decoder: _Decoder = _Decoder()

        self.lr = lr
        self.loss_enum = loss
        self.loss = self._get_loss_from_enum(loss)
        self.test_losses = []

        self.save_hyperparameters(ignore=["decoder", "encoder"])

    def forward(self, x):
        return self.decoder(x)

    def encode(self, x):
        # return self.encoder(x)
        with torch.no_grad():
            _placeholder = torch.zeros(x.size(0), 1).to(x.device)
            _, _, _, _, z, _ = self.encoder(x, _placeholder)
        return z.detach()

    def training_step(self, batch, batch_idx):
        (x, label) = batch
        z = self.encode(x)
        x_reconstructed = self.decoder(z)

        loss = self.loss(x_reconstructed, x)

        self.log(f"Decoder {self.loss_enum}/train_loss", loss, batch_size=x.size(0))
        return loss

    # validation step
    def validation_step(self, batch, batch_idx):
        (x, label) = batch
        z = self.encode(x)

        x_reconstructed = self.decoder(z)
        loss = self.loss(x_reconstructed, x)
        self.log(f"Decoder {self.loss_enum}/val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        (x, label) = batch
        z = self.encode(x)
        x_reconstructed = self.decoder(z)

        loss = self.loss(x_reconstructed, x)
        self.test_losses.append(loss)
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_losses).mean()
        self.log(f"Decoder {self.loss_enum}/avg_test_loss", avg_loss)
        self.test_losses = []  # reset for the next epoch

    @staticmethod
    def _get_loss_from_enum(loss_enum: DecoderLoss):
        if loss_enum == DecoderLoss.MSE:
            return MSE_Loss()
        else:
            raise ValueError(f"Loss enum {loss_enum} not supported. In vision project only MSE is supported.")


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: (batch_size, 128, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (batch_size, 64, 28, 28)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (batch_size, 32, 56, 56)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # Output: (batch_size, 3, 112, 112)
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((64, 64))  # Output: (batch_size, 3, 64, 64)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    pass
