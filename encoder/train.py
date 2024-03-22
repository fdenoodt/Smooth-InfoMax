# Example usage:
# xxxx dataset:
# python -m encoder.train temp sim_audio_xxxx_distr_true  --overrides encoder_config.dataset.dataset=4 encoder_config.dataset.batch_size=64 encoder_config.kld_weight=0.01 encoder_config.num_epochs=10 syllables_classifier_config.encoder_num=9 syllables_classifier_config.dataset.batch_size=64

# for cpc: cpc_audio_xxxx

import os
import torch
import time
import numpy as np
import random
import gc

from config_code.config_classes import OptionsConfig, Dataset, ModelType
from models.full_model import FullModel
from post_hoc_analysis.interpretability.main_anal_hidd_repr import run_visualisations

# own modules
from utils import logger
from arg_parser import arg_parser
from models import load_audio_model
from data import get_dataloader
from utils.utils import set_seed
from validation.val_by_syllables import val_by_latent_syllables
from validation.val_by_InfoNCELoss import val_by_InfoNCELoss

import wandb


def train(opt: OptionsConfig, logs, model: FullModel, optimizer, train_loader, test_loader):
    '''Train the model'''
    total_step = len(train_loader)
    limit_train_batches = opt.encoder_config.dataset.limit_train_batches  # value between 0 and 1
    if limit_train_batches < 1:
        print(f"\nLimiting training to {int(limit_train_batches * 100)}% of the dataset!!!!")
        print(
            f"Limiting validation to {int(opt.encoder_config.dataset.limit_validation_batches * 100)}% of the dataset!!!! \n")
        total_step = int(total_step * limit_train_batches)

    # how often to output training values
    print_idx = 100
    # how often to validate training process by plotting latent representations of various speakers
    latent_val_idx = 1000

    starttime = time.time()

    decay_rate = opt.encoder_config.decay_rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    start_epoch = opt.encoder_config.start_epoch
    num_epochs = opt.encoder_config.num_epochs
    global_step = 0
    for epoch in range(start_epoch, num_epochs + start_epoch):

        nb_modules = len(opt.encoder_config.architecture.modules)
        loss_epoch = [0 for _ in range(nb_modules)]

        for step, (audio, _, _, _) in enumerate(train_loader):

            # validate training progress by plotting latent representation of various speakers
            # TODO
            # if step % latent_val_idx == 0 and opt.encoder_config.dataset.dataset == Dataset.xxxx:
            #     val_by_latent_syllables(opt.encoder_config.dataset, opt.device, test_loader, model, epoch, step)

            if step % print_idx == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs + start_epoch}], Step [{step}/{total_step}], Time (s): {time.time() - starttime:.1f}"
                )

            starttime = time.time()

            # shape: (batch_size, 1, 8800)
            model_input = audio.to(opt.device)
            loss, nce, kld = model(model_input)  # loss for each module

            # Average over the losses from different GPUs
            loss = torch.mean(loss, 0)
            nce = torch.mean(nce, 0)
            kld = torch.mean(kld, 0)

            model.zero_grad()
            overall_loss = sum(loss)
            overall_loss.backward()
            optimizer.step()

            for idx, cur_losses in enumerate(loss):
                print_loss = cur_losses.item()
                loss_epoch[idx] += print_loss

                if step % print_idx == 0:
                    print(f"\t \t Loss: \t \t {print_loss:.4f}")

            for idx, cur_nce in enumerate(nce):
                wandb.log({f"nce_{idx}": cur_nce}, step=global_step)
            for idx, cur_kld in enumerate(kld):
                wandb.log({f"kld_{idx}": cur_kld}, step=global_step)
            for idx, cur_losses in enumerate(loss):
                wandb.log({f"loss_{idx}": cur_losses}, step=global_step)

            wandb.log({'epoch': epoch}, step=global_step)

            global_step += 1

            if step >= total_step:
                break

        scheduler.step()
        print(f"LR: {scheduler.get_last_lr()}")

        logs.append_train_loss([x / total_step for x in loss_epoch])

        # validate by testing the CPC performance on the validation set
        if opt.validate:
            validation_loss = val_by_InfoNCELoss(opt, model, test_loader)
            logs.append_val_loss(validation_loss)

            for i, val_loss in enumerate(validation_loss):
                wandb.log({f"val_loss_{i}": val_loss}, step=global_step)

        if (epoch % opt.log_every_x_epochs == 0):
            logs.create_log(model, optimizer=optimizer, epoch=epoch)


def _main(options: OptionsConfig):
    if options.encoder_config.architecture.is_cpc:
        family = "CPC"
    elif options.encoder_config.architecture.modules[0].predict_distributions:
        family = "SIM"
    else:
        family = "GIM"
    run_name = f"{family}_kld={options.encoder_config.kld_weight}_lr={options.encoder_config.learning_rate}_{int(time.time())}"

    project_name = "SIM_ENCODER_FULL_PIPELINE_512dim_x_256dim_repeat_NO_SEED_v2"
    wandb.init(project=project_name, name=run_name)
    for key, value in vars(options).items():
        wandb.config[key] = value

    # After initializing the wandb run, get the run id
    run_id = wandb.run.id
    # Save the run id to a file in the logs directory
    with open(os.path.join(options.log_path, 'wandb_run_id.txt'), 'w') as f:
        f.write(run_id)
        # write project name to file
        f.write(f"\n{project_name}")

    options.model_type = ModelType.ONLY_ENCODER
    logs = logger.Logger(options)

    assert options.model_type == ModelType.ONLY_ENCODER, "Only encoder training is supported."

    # load model
    model, optimizer = load_audio_model.load_model_and_optimizer(options, None)

    # get datasets and dataloaders
    train_loader, train_dataset, test_loader, test_dataset = get_dataloader.get_dataloader(
        config=options.encoder_config.dataset)

    try:
        # Train the model
        train(options, logs, model, optimizer, train_loader, test_loader)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)

    wandb.finish()

    # TODO:
    # Save_latents_and_generate_visualisations(options)


def _init(options: OptionsConfig):
    torch.cuda.empty_cache()
    gc.collect()

    arg_parser.create_log_path(options)

    # set random seeds
    set_seed(options.seed)



def run_configuration(options: OptionsConfig):
    _init(options)
    _main(options)


if __name__ == "__main__":
    from options import get_options

    run_configuration(get_options())
