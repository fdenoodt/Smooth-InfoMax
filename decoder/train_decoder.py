import time

import numpy as np
import torch

from decoder.decoderr import Decoder
from decoder.lit_decoder import LitDecoder
from utils import logger

from arg_parser import arg_parser
from config_code.config_classes import OptionsConfig, ModelType, Dataset
from data import get_dataloader
from decoder.callbacks import CustomCallback
from decoder.my_data_module import MyDataModule

from models import load_audio_model
from models.full_model import FullModel
from options import get_options

import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


def main(model_type: ModelType = ModelType.ONLY_DOWNSTREAM_TASK):
    torch.set_float32_matmul_precision('medium')

    opt: OptionsConfig = get_options()
    print(f"\nTRAINING DECODER USING LOSS: {opt.decoder_config.decoder_loss} \n")

    opt.model_type = model_type

    decoder_config = opt.decoder_config

    assert opt.decoder_config is not None, "Decoder config is not set"
    assert opt.model_type in [ModelType.ONLY_DOWNSTREAM_TASK], "Model type not supported"
    assert (opt.decoder_config.dataset.dataset in
            [Dataset.DE_BOER_RESHUFFLED, Dataset.DE_BOER_RESHUFFLED_V2]), "Dataset not supported"

    arg_parser.create_log_path(opt, add_path_var="decoder_model")

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    distr: bool = opt.encoder_config.architecture.modules[0].predict_distributions
    wandb.init(project="SIM_DECODER",
               name=f"[distr={distr}_kld={opt.encoder_config.kld_weight}]_l={opt.decoder_config.decoder_loss}_lr={opt.decoder_config.learning_rate}" +
                    f"_{int(time.time())}",
               tags=[f"distr={distr}", f"kld={opt.encoder_config.kld_weight}", f"l={opt.decoder_config.decoder_loss}",
                     f"lr={opt.decoder_config.learning_rate}"]
               )

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
    callback = CustomCallback(opt, z_dim=z_dim, wandb_logger=wandb_logger, nb_frames=nb_frames, plot_ever_n_epoch=2)
    trainer = L.Trainer(limit_train_batches=decoder_config.dataset.limit_train_batches,
                        max_epochs=decoder_config.num_epochs,
                        accelerator="gpu", devices="1",
                        logger=wandb_logger, callbacks=[callback])
    trainer.fit(model=lit, datamodule=data)
    trainer.test(model=lit, datamodule=data)

    # arg_parser.create_log_path(opt, add_path_var="linear_model_syllables")
    # logs.create_log(loss, accuracy=accuracy, final_test=True, final_loss=result_loss)

    logs.create_log(decoder, final_test=True, final_loss=[])


if __name__ == "__main__":
    main()
