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
import matplotlib.pyplot as plt
import random


if(True):
    importlib.reload(decoder_architectures)
    importlib.reload(helper_functions)

    from decoder_architectures import *
    from helper_functions import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'



random.seed(0)


class LogHandler():
    def __init__(self, logs, train_loader) -> None:
        self.total_step = len(train_loader)
        self.logs: logger.Logger = logs

    def __call__(self, loss_epoch, *args: Any, **kwds: Any) -> Any:
        self.logs.append_train_loss([x / self.total_step for x in loss_epoch])


class EpochPrinter():
    def __init__(self, train_loader) -> None:
        self.starttime = time.time()

        self.print_idx = 100
        self.step = 0
        self.total_step = len(train_loader)

    def __call__(self, step, epoch, *args: Any, **kwds: Any) -> Any:
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


def train(decoder, logs, train_loader):
    # epoch_printer = EpochPrinter(train_loader)
    # log_handler = LogHandler(logs, train_loader)

    # decoder.to(device)
    # decoder.train()

    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(decoder.parameters(), lr=1.5e-2)

    for epoch in range(opt["start_epoch"], opt["num_epochs"] + opt["start_epoch"]):

        loss_epoch = [0]

        for step, (org_audio, enc_audio, _, _, _) in enumerate(train_loader):
            pass
            # epoch_printer(step, epoch)

            # enc_audios = enc_audio.to(device)
            # org_audio = org_audio.to(device)

            # # zero the gradients
            # optimizer.zero_grad()

            # # forward pass
            # outputs = decoder(enc_audios)
            # loss = criterion(outputs, org_audio)

            # # backward pass and optimization step
            # loss.backward()
            # optimizer.step()

            # # print the loss at each step

            # loss_epoch[0] += loss.item()
            # </> end for step

        # log_handler(loss_epoch) # store losses
        print(epoch)
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
    opt['num_epochs'] = 5
    opt['batch_size'] = 8

    logs = logger.Logger(opt)

    # load the data
    train_loader, _, _, _ = get_dataloader.\
        get_de_boer_sounds_decoder_data_loaders(opt)

    # two_layer_decoder = OneLayerDecoder()
    decoder = train(None, logs, train_loader)
    # decoder = train(two_layer_decoder, logs, train_loader)

    # logs.create_log(two_layer_decoder)

    torch.cuda.empty_cache()


# %%
