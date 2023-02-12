# %%
import time
import importlib
from GIM_encoder import GIM_Encoder
import decoder_architectures
import helper_functions
from options import OPTIONS as opt
import torch

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

    print(
        f"Validation Loss: Time (s): {time.time() - starttime:.1f} --- {loss_epoch[0] / total_step:.4f}")

    validation_loss = np.mean(loss_epoch)
    return validation_loss


def train(decoder, logs, train_loader, test_loader):
    epoch_printer = EpochPrinter(train_loader)
    log_handler = LogHandler(opt, logs, train_loader)

    normalize_func = compute_normalizer(train_loader, encoder)


    decoder.to(device)
    decoder.train()

    criterion = nn.MSELoss()
    # criterion = MSE_AND_SPECTRAL_LOSS(128)  # number is adviced by chat gpt
    # criterion = MSE_AND_SPECTRAL_LOSS(8192) # number is adviced by chat gpt
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-5)  # 1.5 * 10^-2 = 1.5/100

    training_losses = []
    validation_losses = []
    for epoch in range(opt["start_epoch"], opt["num_epochs"] + opt["start_epoch"]):

        training_losses_epoch = []
        for step, (ground_truth_audio_batch, _, _, _) in enumerate(train_loader):
            epoch_printer(step, epoch)

            ground_truth_audio_batch = ground_truth_audio_batch.to(device)  # (batch_size, 1, 10240)
            enc_audios = normalize_func(encoder(ground_truth_audio_batch).to(device))  # (batch_size, 512, 256)

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



if __name__ == "__main__":
    torch.cuda.empty_cache()

    arg_parser.create_log_path(opt)

    experiment_name = 'RMSE_decoder_GIM_layer3_MSE_loss'
    # experiment_name = 'RMSE_decoder_GIM_layer3_MSE_SPECTRAL_loss'
    opt['experiment'] = experiment_name
    opt['save_dir'] = f'{experiment_name}_experiment'
    opt['log_path'] = f'./logs/{experiment_name}_experiment'
    opt['log_path_latent'] = f'./logs/{experiment_name}_experiment/latent_space'
    opt['num_epochs'] = 50
    opt['batch_size'] = 64 + 32
    opt['batch_size_multiGPU'] = opt['batch_size']

    create_log_dir(opt['log_path'])

    logs = logger.Logger(opt)

    # load the data
    train_loader, _, test_loader, _ = get_dataloader.\
        get_de_boer_sounds_decoder_data_loaders(opt)

    encoder = GIM_Encoder(opt, layer_depth=3, path="DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/model_360.ckpt")
    two_layer_decoder = TwoLayerDecoder()
    decoder = train(two_layer_decoder, logs, train_loader, test_loader)

    torch.cuda.empty_cache()

    # %%
