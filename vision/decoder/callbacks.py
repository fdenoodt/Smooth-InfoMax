import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from vision.decoder.decoderr import Decoder
import wandb

import torch
from vision.decoder.decoderr import Decoder
from lightning.pytorch.loggers import WandbLogger
# from vision.decoder.callbacks import CustomCallback
from config_code.config_classes import OptionsConfig, ModelType, DecoderLoss
from vision.models import load_vision_model
from vision.data import get_dataloader
from decoder.my_data_module import MyDataModule
from options import get_options
from utils.utils import set_seed
from lightning import Trainer


class CustomCallback(L.Callback):
    def __init__(self, z_dim, test_loader, z_height=7, z_width=7):
        super().__init__()
        self.z_dim = z_dim
        self.z_height = z_height
        self.z_width = z_width
        self.test_loader = test_loader
        self.log_every_n_epoch = 10
        self.nb_samples = 10

    def _random_samples(self, decoder: Decoder, num_samples):
        z_batch = torch.randn(num_samples, self.z_dim, self.z_height, self.z_width).to(decoder.device)
        x_reconstructed = decoder.forward(z_batch)
        return x_reconstructed

    def _reconstruction(self, decoder: Decoder, x):
        z = decoder.encode(x)
        x_reconstructed = decoder.forward(z)
        return x_reconstructed

    def log_images(self, trainer: L.Trainer, decoder: Decoder, when: str):
        nb_samples = self.nb_samples

        # Reconstruction
        x, _ = next(iter(self.test_loader))
        x = x.to(decoder.device)
        x_reconstructed = self._reconstruction(decoder, x)

        # Random samples
        x_samples = self._random_samples(decoder, nb_samples)

        # Log images
        for i in range(nb_samples):
            image_x = wandb.Image(
                x[i].cpu().detach().numpy().transpose((1, 2, 0)),
                caption="Original image")
            image_x_reconstructed = wandb.Image(
                x_reconstructed[i].cpu().detach().numpy().transpose((1, 2, 0)),
                caption="Reconstructed image")
            image_x_sample = wandb.Image(
                x_samples[i].cpu().detach().numpy().transpose((1, 2, 0)),
                caption="Random sample")

            key_part = f"decoder/{when}"
            wandb.log({
                f"{key_part}/original/{i}": image_x,
                f"{key_part}/reconstructed/{i}": image_x_reconstructed,
                f"{key_part}/samples/{i}": image_x_sample
            }, step=trainer.global_step)

    def on_train_epoch_end(self, trainer: L.Trainer, decoder: Decoder):
        if trainer.current_epoch % self.log_every_n_epoch == 0:
            self.log_images(trainer, decoder, when="on_train_epoch_end")

    def on_train_end(self, trainer: L.Trainer, decoder: Decoder):
        self.log_images(trainer, decoder, when="on_train_end")


def test_callbacks():
    # initialize wandb
    wandb.init(project="test")

    # Set up the options
    opt: OptionsConfig = get_options()
    set_seed(opt.seed)

    # Load the model
    context_model, _ = load_vision_model.load_model_and_optimizer(
        opt, reload_model=True, calc_loss=False, downstream_config=opt.decoder_config
    )
    context_model.eval()

    # Set up the data
    train_loader, _, _, _, test_loader, _ = get_dataloader.get_dataloader(
        opt.encoder_config.dataset,
        purpose_is_unsupervised_learning=True,
    )
    data = MyDataModule(train_loader, test_loader, test_loader)

    # Set up the decoder
    decoder = Decoder(encoder=context_model,
                      lr=opt.decoder_config.learning_rate,
                      loss=opt.decoder_config.decoder_loss).to(opt.device)

    # Set up the callback
    wandb_logger = WandbLogger() if opt.use_wandb else None
    callback = CustomCallback(test_loader=test_loader, z_dim=256)

    # Set up the trainer
    trainer = Trainer(limit_train_batches=opt.decoder_config.dataset.limit_train_batches,
                      max_epochs=opt.decoder_config.num_epochs,
                      logger=wandb_logger, callbacks=[callback])

    # Call the on_train_epoch_end method
    callback.on_train_epoch_end(trainer, decoder)

    # Call the on_train_end method
    callback.on_train_end(trainer, decoder)

    # Finish the test
    wandb.finish()


if __name__ == "__main__":
    test_callbacks()
