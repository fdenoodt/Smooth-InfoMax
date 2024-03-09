import lightning as L
import torch
import torch.nn as nn
from torch import optim

from config_code.config_classes import DecoderLoss
from decoder.decoder_losses import MSE_Loss, SpectralLoss, MSE_AND_SPECTRAL_LOSS, FFTLoss, MSE_AND_FFT_LOSS, MEL_LOSS, \
    MSE_AND_MEL_LOSS, MEL_LOSS
from models.full_model import FullModel


class LitDecoder(L.LightningModule):
    def __init__(self, encoder, decoder: nn.Module, lr: float, loss: DecoderLoss):
        super().__init__()
        encoder.eval()
        self.encoder = encoder

        self.decoder = decoder
        self.lr = lr
        self.loss_enum = loss
        self.loss = self._get_loss_from_enum(loss)
        self.test_losses = []

        self.save_hyperparameters(ignore=["decoder", "encoder"])

    def training_step(self, batch, batch_idx):
        (x, _, label, _) = batch  # x.shape: (200, 1, 64, 64), y.shape: (200, 1, 6)
        with torch.no_grad():
            full_model: FullModel = self.encoder.module
            z = full_model.forward_through_all_cnn_modules(x)
        z = z.detach()
        x_reconstructed = self.decoder(z)

        loss = self.loss(x_reconstructed, x)

        self.log(f"Decoder {self.loss_enum}/train_loss", loss, batch_size=x.size(0))
        return loss

    # validation step
    def validation_step(self, batch, batch_idx):
        (x, _, label, _) = batch
        with torch.no_grad():
            full_model: FullModel = self.encoder.module
            z = full_model.forward_through_all_cnn_modules(x)
        z = z.detach()

        x_reconstructed = self.decoder(z)
        loss = self.loss(x_reconstructed, x)
        self.log(f"Decoder {self.loss_enum}/val_loss", loss)
        return loss

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
        self.log(f"Decoder {self.loss_enum}/avg_test_loss", avg_loss)
        self.test_losses = []  # reset for the next epoch

    @staticmethod
    def _get_loss_from_enum(loss_enum: DecoderLoss):
        if loss_enum == DecoderLoss.MSE:
            return MSE_Loss()
        elif loss_enum == DecoderLoss.SPECTRAL:
            return SpectralLoss()
        elif loss_enum == DecoderLoss.MSE_SPECTRAL:
            return MSE_AND_SPECTRAL_LOSS()
        elif loss_enum == DecoderLoss.FFT:
            return FFTLoss()
        elif loss_enum == DecoderLoss.MSE_FFT:
            return MSE_AND_FFT_LOSS()
        elif loss_enum == DecoderLoss.MEL:
            return MEL_LOSS()
        elif loss_enum == DecoderLoss.MSE_MEL:
            return MSE_AND_MEL_LOSS()
        else:
            raise ValueError(f"Loss enum {loss_enum} not supported")


if __name__ == "__main__":
    pass
