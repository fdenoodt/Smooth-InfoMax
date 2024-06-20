# Example usage:
# python -m decoder.train_decoder temp sim_audio_de_boer_distr_true
# --overrides decoder_config.decoder_loss=0 decoder_config.encoder_num=9
from typing import Dict

import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from arg_parser import arg_parser
from config_code.architecture_config import DecoderArchitectureConfig
from config_code.config_classes import OptionsConfig, ModelType, Dataset, DecoderLoss, DecoderConfig
from data import get_dataloader
from decoder.callbacks import CustomCallback
from decoder.decoderr import Decoder
from decoder.lit_decoder import LitDecoder
from decoder.my_data_module import MyDataModule
from models import load_audio_model
from options import get_options
from utils import logger
from utils.utils import retrieve_existing_wandb_run_id, set_seed, get_audio_decoder_key


def main(model_type: ModelType = ModelType.ONLY_DOWNSTREAM_TASK):
    torch.set_float32_matmul_precision('medium')

    opt: OptionsConfig = get_options()
    decoder_config: DecoderConfig = opt.decoder_config
    loss_fun: DecoderLoss = decoder_config.decoder_loss
    print(f"\nTRAINING DECODER USING LOSS: {loss_fun} \n")

    opt.model_type = model_type

    assert opt.decoder_config is not None, "Decoder config is not set"
    assert opt.model_type in [ModelType.ONLY_DOWNSTREAM_TASK], "Model type not supported"
    assert (decoder_config.dataset.dataset in [Dataset.DE_BOER]), "Dataset not supported"

    # get integer val of enum
    loss_val = loss_fun.value  # eg 0 for DecoderLoss.MSE

    # random seeds
    set_seed(opt.seed)

    if opt.use_wandb:
        # run_id, project_name = retrieve_existing_wandb_run_id(opt)
        # wandb.init(id=run_id, resume="allow", project=project_name)

        wandb.init(project="temp")

    # MUST HAPPEN AFTER wandb.init
    key = get_audio_decoder_key(decoder_config, loss_val)  # for path and wandb section
    arg_parser.create_log_path(opt, key)

    wandb_logger = WandbLogger() if opt.use_wandb else None

    context_model, _ = load_audio_model.load_model_and_optimizer(
        opt,
        decoder_config,
        reload_model=True,
        calc_accuracy=False,  # True,
        num_GPU=1,
    )
    context_model.eval()

    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(decoder_config.dataset)
    data = MyDataModule(train_loader, test_loader, test_loader)

    architecture: DecoderArchitectureConfig = DecoderConfig.retrieve_correct_decoder_architecture(decoder_config)

    decoder = Decoder(architecture)

    logs = logger.Logger(opt)

    lit = LitDecoder(decoder_config,
                     context_model, decoder,
                     decoder_config.learning_rate,
                     decoder_config.decoder_loss)

    z_dim = architecture.input_dim
    nb_frames = architecture.expected_nb_frames_latent_repr
    callback = CustomCallback(opt, z_dim=z_dim, wandb_logger=wandb_logger, nb_frames=nb_frames,
                              plot_ever_n_epoch=10, loss_enum=loss_fun) if opt.use_wandb else None

    trainer = L.Trainer(limit_train_batches=decoder_config.dataset.limit_train_batches,
                        max_epochs=decoder_config.num_epochs,
                        accelerator="gpu", devices="1",
                        log_every_n_steps=10,  # arbitrary number to avoid warning
                        logger=wandb_logger, callbacks=[callback] if callback is not None else [])

    if opt.train:
        trainer.fit(model=lit, datamodule=data)

    trainer.test(model=lit, datamodule=data)

    # The following line doesn't overwrite the last encoder (stores to adjusted log path)
    # which was done in `arg_parser.create_log_path()`
    logs.create_log(decoder, final_test=True, final_loss=[])

    if opt.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
