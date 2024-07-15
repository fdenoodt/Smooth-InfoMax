# Example usage:
# python -m encoder.train temp sim_audio_de_boer_distr_true --overrides encoder_config.kld_weight=0.01 encoder_config.num_epochs=2 syllables_classifier_config.encoder_num=1 use_wandb=False train=True
# for cpc: cpc_audio_de_boer

import gc

import lightning as L
import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from arg_parser import arg_parser
from config_code.config_classes import OptionsConfig, ModelType
from decoder.my_data_module import MyDataModule
from models import load_audio_model
from utils import logger
from utils.decorators import timer_decorator, wandb_decorator
from utils.utils import set_seed


class ContrastiveModel(L.LightningModule):
    def __init__(self, options: OptionsConfig):
        super(ContrastiveModel, self).__init__()
        self.options = options
        self.model, self.optimizer = load_audio_model.load_model_and_optimizer(options, None)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        audio, _, _, _ = batch
        model_input = audio.to(self.options.device)
        loss, nce, kld = self.model(model_input)

        # Average over the losses from different GPUs
        loss = torch.mean(loss, 0)
        nce = torch.mean(nce, 0)
        kld = torch.mean(kld, 0)

        # zip the losses together
        for idx, (cur_losses, cur_nce, cur_kld) in enumerate(zip(loss, nce, kld)):
            self.log(f'train/loss_{idx}', cur_losses, batch_size=self.options.encoder_config.dataset.batch_size)
            self.log(f'nce/nce_{idx}', cur_nce, batch_size=self.options.encoder_config.dataset.batch_size)
            self.log(f'kld/kld_{idx}', cur_kld, batch_size=self.options.encoder_config.dataset.batch_size)
        return loss.sum()  # sum of all module losses

    def validation_step(self, batch, batch_idx):
        audio, _, _, _ = batch
        model_input = audio.to(self.options.device)
        loss, nce, kld = self.model(model_input)

        # Average over the losses from different GPUs
        loss = torch.mean(loss, 0)

        for i, modul_loss in enumerate(loss):
            self.log(f'validation/val_loss_{i}', modul_loss, batch_size=self.options.encoder_config.dataset.batch_size)
        return loss.sum()  # sum of all module losses

    def configure_optimizers(self):
        return [self.optimizer]


@timer_decorator
@wandb_decorator  # calls wandb.init
# @profile_decorator  # calls torch.profiler.profile
def _main(options: OptionsConfig):
    options.model_type = ModelType.ONLY_ENCODER
    logs = logger.Logger(options)

    assert options.model_type == ModelType.ONLY_ENCODER, \
        "Only encoder training is supported."

    model = ContrastiveModel(options)
    if options.compile_model:
        try:
            model = torch.compile(model, mode='default')  # Compile it to make it faster
        except Exception as e:
            print(f"Could not compile model: {e}")

    data_module = MyDataModule(options.encoder_config.dataset)

    trainer = Trainer(
        max_epochs=options.encoder_config.num_epochs,
        limit_train_batches=options.encoder_config.dataset.limit_train_batches,
        limit_val_batches=options.encoder_config.dataset.limit_validation_batches,
        logger=WandbLogger() if options.use_wandb else None,
        # callbacks=[profiler_callback] if options.profile else None,
        log_every_n_steps=10
    )

    if options.train:
        try:
            # Train the model
            trainer.fit(model, data_module)
        except KeyboardInterrupt:
            print("Training got interrupted, saving log-files now.")

    logs.create_log(model)


def _init(options: OptionsConfig):
    torch.set_float32_matmul_precision('medium')
    torch.cuda.empty_cache()
    gc.collect()

    arg_parser.create_log_path(options)

    # set random seeds
    set_seed(options.seed)


if __name__ == "__main__":
    from options import get_options

    _options = get_options()

    print("*" * 80)
    print(_options)
    print("*" * 80)
    print()

    _init(_options)
    _main(_options)
