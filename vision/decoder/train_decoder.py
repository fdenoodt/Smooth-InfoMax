# Example usage:
# python -m decoder.train_decoder vis_dir vision_default --overrides encoder_config.dataset.dataset=9 vision_classifier_config.dataset.dataset=7 encoder_config.num_epochs=200 encoder_config.dataset.grayscale=False vision_classifier_config.dataset.grayscale=False encoder_config.architecture.resnet_type=34 decoder_config.encoder_num=0 use_wandb=False train=True

import time

import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from arg_parser import arg_parser
from config_code.config_classes import OptionsConfig, ModelType, Dataset, DecoderLoss
from vision.data import get_dataloader
from vision.decoder.callbacks import CustomCallback
from vision.decoder.decoderr import Decoder
from decoder.my_data_module import MyDataModule
from options import get_options
from utils import logger
from utils.utils import retrieve_existing_wandb_run_id, set_seed
from vision.models import load_vision_model


def main(model_type: ModelType = ModelType.ONLY_DOWNSTREAM_TASK):
    torch.set_float32_matmul_precision('medium')

    opt: OptionsConfig = get_options()
    TRAIN = opt.train
    USE_WANDB = opt.use_wandb
    loss_fun: DecoderLoss = opt.decoder_config.decoder_loss
    print(f"\nTRAINING DECODER USING LOSS: {loss_fun} \n")

    opt.model_type = model_type

    decoder_config = opt.decoder_config

    assert opt.decoder_config is not None, "Decoder config is not set"
    assert opt.model_type in [ModelType.ONLY_DOWNSTREAM_TASK], "Model type not supported"
    assert (opt.decoder_config.dataset.dataset in [Dataset.SHAPES_3D, Dataset.SHAPES_3D_SUBSET,
                                                   Dataset.STL10]), "Dataset not supported"

    # get integer val of enum
    loss_val = loss_fun.value  # eg 0 for DecoderLoss.MSE

    # random seeds
    set_seed(opt.seed)

    distr: bool = opt.encoder_config.architecture.predict_distributions

    if USE_WANDB:
        run_id, project_name = retrieve_existing_wandb_run_id(opt)
        if run_id is not None:
            # Initialize a wandb run with the same run id
            wandb.init(id=run_id, resume="allow", project=project_name)

    # MUST HAPPEN AFTER wandb.init
    arg_parser.create_log_path(opt, add_path_var=f"decoder_model_l={loss_val}")
    logs = logger.Logger(opt)

    wandb_logger = WandbLogger() if USE_WANDB else None

    context_model, _ = load_vision_model.load_model_and_optimizer(
        opt, reload_model=True, calc_loss=False, downstream_config=opt.decoder_config
    )
    context_model.eval()

    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt.encoder_config.dataset,
        purpose_is_unsupervised_learning=True,
    )
    data = MyDataModule(train_loader, test_loader, test_loader)

    decoder = Decoder(encoder=context_model,
                      lr=decoder_config.learning_rate,
                      loss=loss_fun)

    callback = CustomCallback()

    trainer = L.Trainer(limit_train_batches=decoder_config.dataset.limit_train_batches,
                        max_epochs=decoder_config.num_epochs,
                        logger=wandb_logger, callbacks=[callback])
    if TRAIN:
        trainer.fit(model=decoder, datamodule=data)

    trainer.test(model=decoder, datamodule=data)

    logs.create_log(decoder, final_test=True, final_loss=[])


if __name__ == "__main__":
    main()

    ### Simple test to check if encoder and decoder are working together
    # opt: OptionsConfig = get_options()
    # context_model, _ = load_vision_model.load_model_and_optimizer(
    #     opt, reload_model=True, calc_loss=False, downstream_config=opt.decoder_config
    # )
    # model: Decoder = Decoder(encoder=context_model,
    #                          lr=opt.decoder_config.learning_rate,
    #                          loss=opt.decoder_config.decoder_loss).to(opt.device)
    #
    # rnd_ims = torch.rand((33, 3, 64, 64), device=opt.device)  # 33 images, 3 channels, 64x64
    # z = model.encode(rnd_ims)
    # print(z.shape)
    # x_reconstructed = model.forward(z)
    # print(x_reconstructed.shape)
