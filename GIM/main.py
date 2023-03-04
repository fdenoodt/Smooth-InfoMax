# %%
import torch
import time
import numpy as np
import random
import gc
from options import OPTIONS

# own modules
from utils import logger
from arg_parser import arg_parser
from models import load_audio_model
from data import get_dataloader
from validation.val_by_syllables import val_by_latent_syllables
from validation.val_by_InfoNCELoss import val_by_InfoNCELoss


def train(opt, model, optimizer, train_loader, test_loader):
    '''Train the model'''
    total_step = len(train_loader)

    # how often to output training values
    print_idx = 100
    # how often to validate training process by plotting latent representations of various speakers
    latent_val_idx = 1000

    starttime = time.time()

    for epoch in range(opt["start_epoch"], opt["num_epochs"] + opt["start_epoch"]):

        loss_epoch = [0 for _ in range(opt["model_splits"])]

        for step, (audio, _, _, _) in enumerate(train_loader):
            if(opt["dont_train"]):
                if step == 400:
                    break

            # validate training progress by plotting latent representation of various speakers
            if step % latent_val_idx == 0:
                val_by_latent_syllables(opt, test_loader, model, epoch, step)

            if step % print_idx == 0:
                print(
                    f"Epoch [{epoch + 1}/{opt['num_epochs'] + opt['start_epoch']}], Step [{step}/{total_step}], Time (s): {time.time() - starttime:.1f}"
                )

            starttime = time.time()

            # shape: (batch_size, 1, 8800)
            model_input = audio.to(opt["device"])
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

        logs.append_train_loss([x / total_step for x in loss_epoch])

        # validate by testing the CPC performance on the validation set
        if opt["validate"]:
            validation_loss = val_by_InfoNCELoss(opt, model, test_loader)
            logs.append_val_loss(validation_loss)

        if(epoch % opt['log_every_x_epochs'] == 0):
            logs.create_log(model, epoch=epoch, optimizer=optimizer)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()

    arg_parser.create_log_path(OPTIONS)

    # set random seeds
    torch.manual_seed(OPTIONS["seed"])
    torch.cuda.manual_seed(OPTIONS["seed"])
    np.random.seed(OPTIONS["seed"])
    random.seed(OPTIONS["seed"])

    # load model
    MODEL, OPTIMIZER = load_audio_model.load_model_and_optimizer(OPTIONS)

    # initialize logger
    logs = logger.Logger(OPTIONS)

    # get datasets and dataloaders
    TRAIN_LOADER, TRAIN_DATASET, TEST_LOADER, TEST_DATASET = get_dataloader.get_de_boer_sounds_data_loaders(
        OPTIONS
    )

    try:
        # Train the model
        train(OPTIONS, MODEL, OPTIMIZER, TRAIN_LOADER, TEST_LOADER)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(MODEL)


# %%
