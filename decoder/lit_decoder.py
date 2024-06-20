import lightning as L
import torch
import torch.nn as nn
from torch import optim

from config_code.config_classes import DecoderLoss, DecoderConfig
from decoder.decoder_losses import MSE_Loss, SpectralLoss, MSE_AND_SPECTRAL_LOSS, FFTLoss, MSE_AND_FFT_LOSS, MEL_LOSS, \
    MSE_AND_MEL_LOSS, MEL_LOSS
from decoder.decoderr import Decoder
from models.full_model import FullModel
from utils.utils import get_audio_decoder_key


class LitDecoder(L.LightningModule):
    def __init__(self, opt: DecoderConfig, encoder, decoder: Decoder, lr: float, loss: DecoderLoss):
        super().__init__()
        encoder.eval()
        self.dec_opt: DecoderConfig = opt
        self.encoder = encoder

        self.decoder = decoder
        self.lr = lr
        self.loss_enum = loss
        self.loss = self._get_loss_from_enum(loss)
        self.test_losses = []

        self.save_hyperparameters(ignore=["decoder", "encoder", "opt"])

        # used to do a sanity check later
        self.expected_nb_frames_latent_repr = \
            self.dec_opt.retrieve_correct_decoder_architecture().expected_nb_frames_latent_repr

    def encode(self, x):
        full_model: FullModel = self.encoder.module

        modul_idx = self.dec_opt.encoder_module
        layer_idx = self.dec_opt.encoder_layer

        with torch.no_grad():
            if layer_idx == -1:  # final layer of specified module
                z = full_model.forward_through_module(x, modul_idx)
            else:  # specific layer of specified module
                z = full_model.forward_through_layer(x, modul_idx, layer_idx)

        # Sanity check
        _, _, nb_frames = z.shape
        assert nb_frames == self.expected_nb_frames_latent_repr, \
            (f"Expected {self.expected_nb_frames_latent_repr} frames, got {nb_frames} frames. "
             f"Reconsider decoder_config.encoder_module and decoder_config.encoder_layer provided in config.")

        return z.detach()

    def training_step(self, batch, batch_idx):
        (x, _, label, _) = batch  # x.shape: (200, 1, 64, 64), y.shape: (200, 1, 6)
        z = self.encode(x)
        x_reconstructed = self.decoder(z)

        loss = self.loss(x_reconstructed, x)

        section = get_audio_decoder_key(self.dec_opt, self.loss_enum)
        self.log(f"{section}/train_loss", loss, batch_size=x.size(0))
        return loss

    # validation step
    def validation_step(self, batch, batch_idx):
        (x, _, label, _) = batch
        z = self.encode(x)

        x_reconstructed = self.decoder(z)
        loss = self.loss(x_reconstructed, x)
        section = get_audio_decoder_key(self.dec_opt, self.loss_enum)
        self.log(f"{section}/val_loss", loss, batch_size=x.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        (x, _, label, _) = batch
        z = self.encode(x)
        x_reconstructed = self.decoder(z)

        loss = self.loss(x_reconstructed, x)
        self.test_losses.append(loss)
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        section = get_audio_decoder_key(self.dec_opt, self.loss_enum)
        avg_loss = torch.stack(self.test_losses).mean()
        self.log(f"{section}/avg_test_loss", avg_loss)
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
