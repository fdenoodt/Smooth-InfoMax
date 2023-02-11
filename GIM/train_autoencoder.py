# %%
import os
import time
import importlib
from typing import Any
from GIM_encoder import GIM_Encoder
import decoder_architectures
import helper_functions
import torch.nn as nn
from options import OPTIONS as opt
import torch
from utils import logger
from arg_parser import arg_parser
from data import get_dataloader
import numpy as np
import random

random.seed(0)

if(True):
    importlib.reload(decoder_architectures)
    importlib.reload(helper_functions)

    from decoder_architectures import *
    from helper_functions import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LogHandler():
    def __init__(self, opt, logs, train_loader) -> None:
        self.opt = opt
        self.total_step = len(train_loader)
        self.logs: logger.Logger = logs

    def __call__(self, model, epoch, optimizer, train_loss, val_loss) -> None:
        self.save_train_losses(train_loss, val_loss)
        self.save_model(model, epoch, optimizer)
        self.draw_loss_curve(train_loss, val_loss)

    def save_train_losses(self, train_loss, val_loss):
        np.savetxt(f"{self.opt['log_path']}/training_loss.csv", train_loss, delimiter=",")
        np.savetxt(f"{self.opt['log_path']}/validation_loss.csv", val_loss, delimiter=",")
     
    def save_model(self, model, epoch, optimizer) -> None:
        torch.save(model.state_dict(), f'{self.opt["log_path"]}/model_{epoch}.pt')

        


    def draw_loss_curve(self, train_loss, val_loss):
        assert len(train_loss) == len(val_loss)

        lst_iter = np.arange(len(train_loss))
        plt.plot(lst_iter, np.array(train_loss), "-b", label="train loss")

        lst_iter = np.arange(len(val_loss))
        plt.plot(lst_iter, np.array(val_loss), "-r", label="val loss")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(loc="upper right")

        # save image
        plt.savefig(os.path.join(self.opt["log_path"], "loss.png"))
        plt.close()


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


def validation_loss(GIM_encoder, model, test_loader, criterion):
    # based on GIM/ChatGPT
    total_step = len(test_loader)

    loss_epoch = []
    starttime = time.time()

    for step, (org_audio,  _, _, _) in enumerate(test_loader):
        org_audio = org_audio.to(device)
        enc_audio = GIM_encoder(org_audio).to(device)

        with torch.no_grad():
            outputs = model(enc_audio)
            loss = criterion(outputs, org_audio)
            loss = torch.mean(loss, 0)

            loss_epoch.append(loss.data.cpu().numpy())

    print(f"Validation Loss: Time (s): {time.time() - starttime:.1f} --- {loss_epoch[0] / total_step:.4f}")

    validation_loss = np.mean(loss_epoch)
    return validation_loss


def train(decoder, logs, train_loader, test_loader):
    epoch_printer = EpochPrinter(train_loader)
    log_handler = LogHandler(opt, logs, train_loader)

    decoder.to(device)
    decoder.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2, weight_decay=1e-5)  # 1.5 * 10^-2 = 1.5/100

    training_losses = []
    validation_losses = []
    for epoch in range(opt["start_epoch"], opt["num_epochs"] + opt["start_epoch"]):

        training_losses_epoch = []
        for step, (ground_truth_audio_batch, _, _, _) in enumerate(train_loader):

            ground_truth_audio_batch = ground_truth_audio_batch.to(device) # (batch_size, 1, 10240)
            enc_audios = encoder(ground_truth_audio_batch).to(device) # (batch_size, 512, 256)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            output_batch = decoder(enc_audios)
            loss = criterion(output_batch, ground_truth_audio_batch)

            # backward pass and optimization step
            loss.backward()
            optimizer.step()

            training_losses_epoch.append(loss.item())
            # </> end for step

        training_losses.append(np.mean(training_losses_epoch))
        validation_losses.append(validation_loss(encoder, decoder, test_loader, criterion))

        log_handler(decoder, epoch, optimizer, training_losses, validation_losses)
    
    # </> end epoch

    return decoder



def create_log_dir(path):  # created via chat gpt
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    arg_parser.create_log_path(opt)

    experiment_name = 'RMSE_decoder_GIM_layer3'
    opt['experiment'] = experiment_name
    opt['save_dir'] = f'{experiment_name}_experiment'
    opt['log_path'] = f'./logs/{experiment_name}_experiment'
    opt['log_path_latent'] = f'./logs/{experiment_name}_experiment/latent_space'
    opt['num_epochs'] = 20
    opt['batch_size'] = 64

    create_log_dir(opt['log_path'])

    logs = logger.Logger(opt)

    # load the data
    train_loader, _, test_loader, _ = get_dataloader.\
        get_de_boer_sounds_decoder_data_loaders(opt)
    


    encoder = GIM_Encoder(opt, layer_depth=3, path="./g_drive_model/model_180.ckpt")
    two_layer_decoder = TwoLayerDecoder()
    decoder = train(two_layer_decoder, logs, train_loader, test_loader)
 
    torch.cuda.empty_cache()

    # %%

