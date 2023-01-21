# %%
import time
import importlib
from typing import Any
import decoder_architectures
import helper_functions
import torch.nn as nn
from options import OPTIONS as opt
import torch
from utils import logger
from arg_parser import arg_parser
from data import get_dataloader
import random

random.seed(0)

if(True):
    importlib.reload(decoder_architectures)
    importlib.reload(helper_functions)

    from decoder_architectures import *
    from helper_functions import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LogHandler():
    def __init__(self, logs, train_loader) -> None:
        self.total_step = len(train_loader)
        self.logs: logger.Logger = logs

    def __call__(self, model, epoch, optimizer, train_loss_epoch, val_loss_epoch) -> None:
        self.append_loss(train_loss_epoch, val_loss_epoch)
        self.save_model(model, epoch, optimizer)

    def append_loss(self, train_loss, val_loss) -> None:
        self.logs.append_train_loss([x / self.total_step for x in train_loss])
        self.logs.append_val_loss([x / self.total_step for x in val_loss])

    def save_model(self, model, epoch, optimizer) -> None:
        logs.create_log(model, epoch=epoch, optimizer=optimizer)


class EpochPrinter():
    def __init__(self, train_loader) -> None:
        self.starttime = time.time()

        self.print_idx = 100
        self.step = 0
        self.total_step = len(train_loader)

    def __call__(self, step, epoch) -> Any:
        if step % self.print_idx == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}".format(
                    epoch + 1,
                    opt["num_epochs"] + opt["start_epoch"],
                    step,
                    self.total_step,
                    time.time() - self.starttime,
                )
            )


def validation_loss(model, test_loader, criterion):
    # based on GIM/ChatGPT
    total_step = len(test_loader)

    loss_epoch = [0]
    starttime = time.time()

    for step, (org_audio, enc_audio, _, _, _) in enumerate(test_loader):
        enc_audio = enc_audio.to(device)
        org_audio = org_audio.to(device)

        with torch.no_grad():
            outputs = model(enc_audio)
            loss = criterion(outputs, org_audio)
            loss = torch.mean(loss, 0)
            loss_epoch += loss.data.cpu().numpy()

    # print(
    #     f"Validation Loss: Time (s): {time.time() - starttime:.1f} --- {loss_epoch[0] / total_step:.4f}")

    validation_loss = [x/total_step for x in loss_epoch]
    return validation_loss


def train(decoder, logs, train_loader, test_loader):
    epoch_printer = EpochPrinter(train_loader)
    log_handler = LogHandler(logs, train_loader)

    decoder.to(device)
    decoder.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1.5e-2)

    for epoch in range(opt["start_epoch"], opt["num_epochs"] + opt["start_epoch"]):

        train_loss_epoch = [0]

        for step, (org_audio, enc_audio, _, _, _) in enumerate(train_loader):
            epoch_printer(step, epoch)

            enc_audios = enc_audio.to(device)
            org_audio = org_audio.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            outputs = decoder(enc_audios)
            loss = criterion(outputs, org_audio)

            # backward pass and optimization step
            loss.backward()
            optimizer.step()

            train_loss_epoch[0] += loss.item()
            # </> end for step

        val_loss_epoch = validation_loss(decoder, test_loader, criterion)
        log_handler(decoder, epoch, optimizer,
                    train_loss_epoch, val_loss_epoch)
    return decoder

# %%


if __name__ == "__main__":
    torch.cuda.empty_cache()

    arg_parser.create_log_path(opt)

    experiment_name = 'RMSE_decoder'
    opt['experiment'] = experiment_name
    opt['save_dir'] = f'{experiment_name}_experiment'
    opt['log_path'] = f'./logs/{experiment_name}_experiment'
    opt['log_path_latent'] = f'./logs/{experiment_name}_experiment/latent_space'
    opt['num_epochs'] = 3
    opt['batch_size'] = 64

    logs = logger.Logger(opt)

    # load the data
    train_loader, _, test_loader, _ = get_dataloader.\
        get_de_boer_sounds_decoder_data_loaders(
            opt, layer_depth=1, GIM_encoder_path="./g_drive_model/model_180.ckpt")

    two_layer_decoder = OneLayerDecoder()
    decoder = train(two_layer_decoder, logs, train_loader, test_loader)

    torch.cuda.empty_cache()


    # %%
    
    # %%
