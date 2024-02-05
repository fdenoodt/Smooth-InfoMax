# %%
import torch
import time
import numpy as np
import random
import gc

from configs.config_classes import OptionsConfig, Dataset
from post_hoc_analysis.main_anal_hidd_repr import run_visualisations

# own modules
from utils import logger
from arg_parser import arg_parser
from models import load_audio_model
from data import get_dataloader
from validation.val_by_syllables import val_by_latent_syllables
from validation.val_by_InfoNCELoss import val_by_InfoNCELoss


def train(opt: OptionsConfig, logs, model, optimizer, train_loader, test_loader):
    '''Train the model'''
    total_step = len(train_loader)

    # how often to output training values
    print_idx = 100
    # how often to validate training process by plotting latent representations of various speakers
    latent_val_idx = 1000

    starttime = time.time()

    decay_rate = opt.encoder_config.decay_rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=decay_rate)

    start_epoch = opt.encoder_config.start_epoch
    num_epochs = opt.encoder_config.num_epochs
    for epoch in range(start_epoch, num_epochs + start_epoch):

        nb_modules = len(opt.encoder_config.architecture.modules)
        loss_epoch = [0 for _ in range(nb_modules)]

        for step, (audio, _, _, _) in enumerate(train_loader):

            # validate training progress by plotting latent representation of various speakers
            if step % latent_val_idx == 0 and opt.encoder_config.dataset.dataset == Dataset.DE_BOER:
                val_by_latent_syllables(opt, test_loader, model, epoch, step)

            if step % print_idx == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs + start_epoch}], Step [{step}/{total_step}], Time (s): {time.time() - starttime:.1f}"
                )

            starttime = time.time()

            # shape: (batch_size, 1, 8800)
            model_input = audio.to(opt.device)
            loss = model(model_input)  # loss for each module

            # Average over the losses from different GPUs
            loss = torch.mean(loss, 0)

            model.zero_grad()
            overall_loss = sum(loss)
            overall_loss.backward()
            optimizer.step()

            for idx, cur_losses in enumerate(loss):
                print_loss = cur_losses.item()
                loss_epoch[idx] += print_loss

                if step % print_idx == 0:
                    print(f"\t \t Loss: \t \t {print_loss:.4f}")

        scheduler.step()
        print(f"LR: {scheduler.get_last_lr()}")

        logs.append_train_loss([x / total_step for x in loss_epoch])

        # validate by testing the CPC performance on the validation set
        if opt.validate:
            validation_loss = val_by_InfoNCELoss(opt, model, test_loader)
            logs.append_val_loss(validation_loss)

        if (epoch % opt.log_every_x_epochs == 0):
            logs.create_log(model, epoch=epoch, optimizer=optimizer)


def save_latents_and_generate_visualisations(opt):
    # TODO
    raise NotImplementedError
    if opt['perform_analysis']:
        options_anal = {
            'LOG_PATH': opt['ANAL_LOG_PATH'],
            'EPOCH_VERSION': opt['ANAL_EPOCH_VERSION'],
            'ONLY_LAST_PREDICTION_FROM_TIME_WINDOW': opt['ANAL_ONLY_LAST_PREDICTION_FROM_TIME_WINDOW'],
            'SAVE_ENCODINGS': opt['ANAL_SAVE_ENCODINGS'],
            'AUTO_REGRESSOR_AFTER_MODULE': opt['ANAL_AUTO_REGRESSOR_AFTER_MODULE'],
            'ENCODER_MODEL_DIR': opt['ANAL_ENCODER_MODEL_DIR'],
            'VISUALISE_LATENT_ACTIVATIONS': opt['ANAL_VISUALISE_LATENT_ACTIVATIONS'],
            'VISUALISE_TSNE': opt['ANAL_VISUALISE_TSNE'],
            'VISUALISE_TSNE_ORIGINAL_DATA': opt['ANAL_VISUALISE_TSNE_ORIGINAL_DATA'],
            'VISUALISE_HISTOGRAMS': opt['ANAL_VISUALISE_HISTOGRAMS']
        }

        run_visualisations(opt, options_anal)


def main(options: OptionsConfig):
    logs = logger.Logger(options)

    # load model
    model, optimizer = load_audio_model.load_model_and_optimizer(options)

    train_w_noise = options.encoder_config.train_w_noise
    assert not train_w_noise, "Noise not supported yet."

    # get datasets and dataloaders
    train_loader, train_dataset, test_loader, test_dataset = get_dataloader.get_dataloader(
        config=options.encoder_config.dataset,
        train_noise=train_w_noise,
        split_and_pad=options.encoder_config.dataset.split_in_syllables)

    try:
        # Train the model
        train(options, logs, model, optimizer, train_loader, test_loader)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)

    # TODO:
    # Save_latents_and_generate_visualisations(options)


def init(options: OptionsConfig):
    torch.cuda.empty_cache()
    gc.collect()

    arg_parser.create_log_path(options)

    # set random seeds
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)


def run_configuration(options: OptionsConfig):
    init(options)
    main(options)
