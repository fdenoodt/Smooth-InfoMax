# Example usage:
# python -m decoder.train_decoder temp sim_audio_xxxx_distr_true --overrides decoder_config.decoder_loss=0 decoder_config.encoder_num=9

import time

import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from arg_parser import arg_parser
from config_code.config_classes import OptionsConfig, ModelType, Dataset, DecoderLoss
from data import get_dataloader
from decoder.callbacks import CustomCallback
from decoder.decoderr import Decoder
from decoder.lit_decoder import LitDecoder
from decoder.my_data_module import MyDataModule
from models import load_audio_model
from options import get_options
from utils import logger
from utils.utils import retrieve_existing_wandb_run_id, set_seed


def main(model_type: ModelType = ModelType.ONLY_DOWNSTREAM_TASK):
    torch.set_float32_matmul_precision('medium')

    opt: OptionsConfig = get_options()
    loss_fun: DecoderLoss = opt.decoder_config.decoder_loss
    print(f"\nTRAINING DECODER USING LOSS: {loss_fun} \n")

    opt.model_type = model_type

    decoder_config = opt.decoder_config

    assert opt.decoder_config is not None, "Decoder config is not set"
    assert opt.model_type in [ModelType.ONLY_DOWNSTREAM_TASK], "Model type not supported"
    assert (opt.decoder_config.dataset.dataset in [Dataset.xxxx]), "Dataset not supported"

    # get integer val of enum
    loss_val = loss_fun.value  # eg 0 for DecoderLoss.MSE

    # random seeds
    set_seed(opt.seed)

    distr: bool = opt.encoder_config.architecture.modules[0].predict_distributions

    run_id, project_name = retrieve_existing_wandb_run_id(opt)
    if run_id is not None:
        # Initialize a wandb run with the same run id
        wandb.init(id=run_id, resume="allow", project=project_name)
    else:
        # Initialize a new wandb run
        wandb.init(project="SIM_DECODERv2",
                   name=f"[distr={distr}_kld={opt.encoder_config.kld_weight}]_l={opt.decoder_config.decoder_loss}_lr={opt.decoder_config.learning_rate}" +
                        f"_{int(time.time())}",
                   # Some tags related to encoder and decoder
                   tags=[f"distr={distr}", f"kld={opt.encoder_config.kld_weight}",
                         f"l={opt.decoder_config.decoder_loss}",
                         f"lr={opt.decoder_config.learning_rate}"])

    # MUST HAPPEN AFTER wandb.init
    arg_parser.create_log_path(opt, add_path_var=f"decoder_model_l={loss_val}")

    wandb_logger = WandbLogger()

    context_model, _ = load_audio_model.load_model_and_optimizer(
        opt,
        decoder_config,
        reload_model=True,
        calc_accuracy=False,  # True,
        num_GPU=1,
    )
    context_model.eval()

    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt.decoder_config.dataset)
    data = MyDataModule(train_loader, test_loader, test_loader)

    z_dim = opt.decoder_config.architecture.input_dim
    nb_frames = 64

    decoder = Decoder(opt.decoder_config.architecture)

    logs = logger.Logger(opt)

    lit = LitDecoder(context_model, decoder, opt.decoder_config.learning_rate, opt.decoder_config.decoder_loss)

    callback = CustomCallback(opt, z_dim=z_dim, wandb_logger=wandb_logger, nb_frames=nb_frames,
                              plot_ever_n_epoch=2, loss_enum=loss_fun)

    trainer = L.Trainer(limit_train_batches=decoder_config.dataset.limit_train_batches,
                        max_epochs=decoder_config.num_epochs,
                        accelerator="gpu", devices="1",
                        logger=wandb_logger, callbacks=[callback])
    trainer.fit(model=lit, datamodule=data)
    trainer.test(model=lit, datamodule=data)

    logs.create_log(decoder, final_test=True, final_loss=[])


if __name__ == "__main__":
    main()
