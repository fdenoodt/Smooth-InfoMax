# Example usage:
# python -m encoder.train temp sim_audio_de_boer_distr_true --overrides encoder_config.kld_weight=0.01 encoder_config.num_epochs=2 syllables_classifier_config.encoder_num=1 use_wandb=False train=True
# for cpc: cpc_audio_de_boer


import lightning as L
import torch
from lightning import Trainer, Callback
from lightning.pytorch.loggers import WandbLogger
from config_code.config_classes import OptionsConfig, ModelType
from decoder.my_data_module import MyDataModule
from models import load_audio_model
from models.full_model import FullModel
from utils import logger
from utils.decorators import timer_decorator, wandb_decorator, init_decorator


class LogCreationCallback(Callback):
    def __init__(self, logs):
        self.logs = logs

    def on_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % 10 == 0:
            self.logs.create_log(pl_module, epoch=current_epoch)


class ContrastiveModel(L.LightningModule):
    def __init__(self, options: OptionsConfig):
        super(ContrastiveModel, self).__init__()
        self.options = options
        self.model, self.optimizer = load_audio_model.load_model_and_optimizer(options, None)
        self.model: FullModel = self.model  # for type hinting

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, label = batch
        loss, nce, kld = self.model(x)

        # Average over the losses from different GPUs
        loss = torch.mean(loss, 0)
        nce = torch.mean(nce, 0)
        kld = torch.mean(kld, 0)

        # zip the losses together
        for idx, (cur_losses, cur_nce, cur_kld) in enumerate(zip(loss, nce, kld)):
            self.log(f'train/loss_{idx}', cur_losses, batch_size=self.options.encoder_dataset.batch_size)
            self.log(f'nce/nce_{idx}', cur_nce, batch_size=self.options.encoder_dataset.batch_size)
            self.log(f'kld/kld_{idx}', cur_kld, batch_size=self.options.encoder_dataset.batch_size)
        return loss.sum()  # sum of all module losses

    def validation_step(self, batch, batch_idx):
        x, label = batch
        loss, nce, kld = self.model(x)

        # Average over the losses from different GPUs
        loss = torch.mean(loss, 0)

        for i, modul_loss in enumerate(loss):
            self.log(f'validation/val_loss_{i}', modul_loss, batch_size=self.options.encoder_dataset.batch_size)
        return loss.sum()  # sum of all module losses

    def configure_optimizers(self):
        return [self.optimizer]


@init_decorator  # sets seed and clears cache etc, create log dir
@wandb_decorator  # calls wandb.init
@timer_decorator
def main(options: OptionsConfig):
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

    data_module = MyDataModule(options.encoder_dataset)

    # from lightning.pytorch.callbacks import DeviceStatsMonitor
    # profiler = SimpleProfiler(dirpath=".", filename="perf_logs")
    log_creation_callback = LogCreationCallback(logs)
    trainer = Trainer(
        max_epochs=options.encoder_config.num_epochs,
        limit_train_batches=options.encoder_dataset.limit_train_batches,
        limit_val_batches=options.encoder_dataset.limit_validation_batches,
        logger=WandbLogger() if options.use_wandb else None,
        log_every_n_steps=10,
        profiler="simple" if options.profile else None,
        callbacks=[log_creation_callback],  # Add your custom callback here
        fast_dev_run=options.fast_dev_run,
        overfit_batches=options.overfit_batches,
        # gpus=options.gpus,
        # num_nodes=options.num_nodes, # for DDP.
        # strategy="ddp",
    )

    if options.train:
        try:
            # Train the model
            trainer.fit(model, data_module)
            logs.create_log(model.model, epoch=options.encoder_config.num_epochs - 1)
        except KeyboardInterrupt:
            print("Training got interrupted, saving log-files now.")

    logs.create_log(model.model, epoch=0)  # also do here for easy debugging


if __name__ == "__main__":
    from options import get_options

    _options = get_options()

    print("*" * 80)
    print(_options)
    print("*" * 80)
    print()

    main(_options)
