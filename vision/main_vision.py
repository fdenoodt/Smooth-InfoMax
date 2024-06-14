# example usage:
# python -m vision.main_vision vis_dir vision_default
# or for Animals_with_Attributes:
# python -m vision.main_vision vis_dir vision_default --overrides encoder_config.dataset.dataset=8 vision_classifier_config.dataset.dataset=8 encoder_config.num_epochs=200

import torch
import time
import numpy as np
import os

from config_code.config_classes import ModelType, OptionsConfig
from vision.models.FullModel import FullVisionModel

#### own modules
from utils import logger
from utils.utils import set_seed, initialize_wandb
from vision.arg_parser import arg_parser
from vision.models import load_vision_model
from vision.data import get_dataloader
from options import get_options

import wandb


def validate(opt: OptionsConfig, model: FullVisionModel, test_loader):
    total_step = len(test_loader)
    model_splits = 3  # TODO
    # model_splits = opt.model_splits

    loss_epoch = [0 for i in range(model_splits)]
    nce_loss_epoch = [0 for i in range(model_splits)]
    kld_loss_epoch = [0 for i in range(model_splits)]
    starttime = time.time()

    for step, (img, label) in enumerate(test_loader):
        model_input = img.to(opt.device)
        label = label.to(opt.device)

        loss, nce_loss, kld_loss, _, _, _ = model(model_input, label, n=opt.encoder_config.architecture.train_module)
        loss = torch.mean(loss, 0)
        nce_loss = torch.mean(nce_loss, 0)
        kld_loss = torch.mean(kld_loss, 0)

        loss_epoch += loss.data.cpu().numpy()
        nce_loss_epoch += nce_loss.data.cpu().numpy()
        kld_loss_epoch += kld_loss.data.cpu().numpy()

    for i in range(opt.encoder_config.architecture.model_splits):
        print(
            "Validation Loss Model {}: Time (s): {:.1f} --- {:.4f}".format(
                i, time.time() - starttime, loss_epoch[i] / total_step
            )
        )

    validation_loss = [x / total_step for x in loss_epoch]
    validation_nce_loss = [x / total_step for x in nce_loss_epoch]
    validation_kld_loss = [x / total_step for x in kld_loss_epoch]

    return validation_loss, validation_nce_loss, validation_kld_loss


def train(opt: OptionsConfig, model: FullVisionModel):
    total_step = len(train_loader)
    model.module.switch_calc_loss(True)

    print_idx = 100

    starttime = time.time()
    # cur_train_module = opt.train_module # TODO
    cur_train_module = 3

    global_step = 0

    for epoch in range(opt.encoder_config.start_epoch, opt.encoder_config.num_epochs + opt.encoder_config.start_epoch):

        model_splits = opt.encoder_config.architecture.model_splits
        loss_epoch = [0 for _ in range(model_splits)]
        loss_updates = [1 for _ in range(model_splits)]

        for step, (img, label) in enumerate(train_loader):

            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Training Block: {}, Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.encoder_config.num_epochs + opt.encoder_config.start_epoch,
                        step,
                        total_step,
                        cur_train_module,
                        time.time() - starttime,
                    )
                )

            starttime = time.time()

            model_input = img.to(opt.device)
            label = label.to(opt.device)

            loss, nce_loss, kld_loss, _, _, accuracy = model(model_input, label, n=cur_train_module)
            loss = torch.mean(loss, 0)  # Take mean over outputs of different GPUs.
            nce_loss = torch.mean(nce_loss, 0)
            kld_loss = torch.mean(kld_loss, 0)
            accuracy = torch.mean(accuracy, 0)

            if cur_train_module != model_splits and model_splits > 1:
                loss = loss[cur_train_module].unsqueeze(0)

            model.zero_grad()
            overall_loss = torch.sum(loss)

            overall_loss.backward()
            optimizer.step()

            for idx, cur_losses in enumerate(loss):
                print_loss = cur_losses.item()
                print_acc = accuracy[idx].item()
                loss_epoch[idx] += print_loss
                loss_updates[idx] += 1

                if step % print_idx == 0:
                    print("\t \t Loss: \t \t {:.4f}".format(print_loss))
                    if opt.loss == 1:
                        print("\t \t Accuracy: \t \t {:.4f}".format(print_acc))

            for idx, cur_losses in enumerate(loss):
                if USE_WANDB:
                    wandb.log(
                        {f"loss_{idx}": cur_losses, f"nce_loss_{idx}": nce_loss[idx], f"kld_loss_{idx}": kld_loss[idx],
                         "epoch": epoch},
                        step=global_step)

            global_step += 1

        if opt.validate:
            validation_loss, validation_nce_loss, validation_kld_loss = \
                validate(opt, model, test_loader)  # Test_loader corresponds to validation set here.

            for i, val_loss in enumerate(validation_loss):
                if USE_WANDB:
                    wandb.log({f"val_loss_{i}": val_loss, f"val_nce_loss_{i}": validation_nce_loss[i],
                               f"val_kld_loss_{i}": validation_kld_loss[i]},
                              step=global_step)

        logs.create_log(model, epoch=epoch, optimizer=optimizer)


if __name__ == "__main__":

    opt: OptionsConfig = get_options()
    USE_WANDB = opt.use_wandb
    TRAIN = opt.train

    assert opt.experiment == "vision"

    dataset = opt.encoder_config.dataset.dataset
    arg_parser.create_log_path(opt)

    if USE_WANDB:
        resnet = opt.encoder_config.architecture.resnet_type

        project_name = f"SIM_VISION_ENCODER_{dataset}{'_resnet34' if resnet == 34 else ''}"
        wandb.init(project=project_name, config=vars(opt))

        initialize_wandb(opt, project_name, run_name=None)

    opt.model_type = ModelType.ONLY_ENCODER

    # random seeds
    set_seed(opt.seed)

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model, optimizer = load_vision_model.load_model_and_optimizer(opt, downstream_config=None)

    logs = logger.Logger(opt)

    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt.encoder_config.dataset,
        purpose_is_unsupervised_learning=True,
    )

    if opt.loss == 1:
        train_loader = supervised_loader

    try:
        # Train the model
        if TRAIN:
            train(opt, model)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)

    if USE_WANDB:
        wandb.finish()

    print("Done!")
