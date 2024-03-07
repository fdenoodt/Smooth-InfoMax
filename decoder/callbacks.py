import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from wandb import Audio

from config_code.config_classes import OptionsConfig
from data import get_dataloader
from decoder.lit_decoder import LitDecoder
from models.full_model import FullModel


class CustomCallback(L.Callback):
    def __init__(self, opt: OptionsConfig, plot_ever_n_epoch, z_dim, nb_frames, wandb_logger: WandbLogger, loss_enum):
        super().__init__()
        self.opt = opt
        self.plot_ever_n_epoch = plot_ever_n_epoch
        self.z_dim = z_dim
        self.nb_frames = nb_frames
        self.wandb_logger = wandb_logger
        self.loss_enum = loss_enum

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: LitDecoder):  # log generated audio (std normal samples)
        # Every 10th epoch, generate some images
        if trainer.current_epoch % self.plot_ever_n_epoch == 0:
            pl_module.eval()

            nb_files = 10

            gen_z = torch.randn((nb_files, self.z_dim, self.nb_frames), requires_grad=False, device=pl_module.device)
            audio_samples = pl_module.decoder(gen_z)  # shape: (100, 1, 2505)
            # remove 1 channel
            audio_samples = audio_samples.squeeze(1)
            audio_samples = audio_samples.contiguous().cpu().data.numpy()

            ten_audio_sammples = [audio_sample for audio_sample in audio_samples[:nb_files]]
            self.wandb_logger.log_audio(key=f"Decoder {self.loss_enum}/std normal samples",
                                        audios=ten_audio_sammples,
                                        sample_rate=[16_000] * nb_files)

            pl_module.train()

    def on_train_end(self, trainer, pl_module):  # log encoded + decoded audio vs gt audio
        pl_module.eval()
        nb_files = 10

        _, _, test_loader, _ = get_dataloader.get_dataloader(self.opt.decoder_config.dataset)
        for i, (audio, _, label, _) in enumerate(test_loader):
            audio = audio.to(pl_module.device)  # shape: (100, 1, 2505)
            with torch.no_grad():
                full_model: FullModel = pl_module.encoder.module
                z = full_model.forward_through_all_cnn_modules(audio)
            z = z.detach()
            x_reconstructed = pl_module.decoder(z)  # shape: (100, 1, 2505)
            x_reconstructed = x_reconstructed.squeeze(1)
            x_reconstructed = x_reconstructed.contiguous().cpu().data.numpy()

            ten_audio_sammples = [audio_sample for audio_sample in x_reconstructed[:nb_files]]
            audio = [audio_sample.squeeze(0).cpu().numpy() for audio_sample in audio[:nb_files]]
            self.wandb_logger.log_audio(
                key=f"Decoder {self.loss_enum}/encode + decode test set",
                audios=ten_audio_sammples, sample_rate=[16_000] * nb_files)

            self.wandb_logger.log_audio(
                key=f"Decoder {self.loss_enum}/gt test set", audios=audio, sample_rate=[16_000] * nb_files)
            break


if __name__ == "__main__":
    import numpy as np

    r = np.zeros(2505)
    r = r.astype(np.float32)
    a = Audio(r, sample_rate=16_000, caption=None)
