# Example usage:
# python -m encoder.train temp sim_audio_de_boer_distr_true --overrides encoder_config.kld_weight=0.01 encoder_config.num_epochs=2 syllables_classifier_config.encoder_num=1 use_wandb=False train=True
# for cpc: cpc_audio_de_boer

import gc
import time

import torch
import wandb

from arg_parser import arg_parser
from config_code.config_classes import OptionsConfig, ModelType
from data import get_dataloader
from models import load_audio_model
from models.full_model import FullModel
# own modules
from utils import logger
from utils.utils import set_seed, initialize_wandb, get_wandb_project_name, timer_decorator
from validation.val_by_InfoNCELoss import val_by_InfoNCELoss


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
            # if step % latent_val_idx == 0 and opt.encoder_config.dataset.dataset == Dataset.DE_BOER:
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

            if opt.use_wandb:
                for idx, cur_nce in enumerate(nce):
                    wandb.log({f"nce/nce_{idx}": cur_nce}, step=global_step)
                for idx, cur_kld in enumerate(kld):
                    wandb.log({f"kld/kld_{idx}": cur_kld}, step=global_step)
                for idx, cur_losses in enumerate(loss):
                    wandb.log({f"loss/loss_{idx}": cur_losses}, step=global_step)

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

            if opt.use_wandb:
                for i, val_loss in enumerate(validation_loss):
                    wandb.log({f"val_loss/val_loss_{i}": val_loss}, step=global_step)

        if (epoch % opt.log_every_x_epochs == 0):
            logs.create_log(model, optimizer=optimizer, epoch=epoch)


@timer_decorator
def _main(options: OptionsConfig):
    if options.use_wandb:
        project_name, run_name = get_wandb_project_name(options)
        initialize_wandb(options, project_name, run_name)

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
        if options.train:
            train(options, logs, model, optimizer, train_loader, test_loader)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)

    if options.use_wandb:
        wandb.finish()


def _init(options: OptionsConfig):
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
