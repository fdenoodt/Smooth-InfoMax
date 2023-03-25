# %%
import time
import importlib
from GIM_encoder import GIM_Encoder
import decoder_architectures
import helper_functions
from options import OPTIONS as opt
from options_autoencoder import OPTIONS as options_autoencoder
import torch

from arg_parser import arg_parser
from data import get_dataloader
import numpy as np
import random

from eval_autoencoder import generate_predictions


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
        enc_audio_per_module = GIM_encoder(org_audio)
        enc_audio = enc_audio_per_module[-1].to(device)
        enc_audio = enc_audio.permute(0, 2, 1)  # (b, c, l)

        with torch.no_grad():
            outputs = model(enc_audio)
            loss = criterion(outputs, org_audio) * \
                (1/org_audio.size(0))  # multiply by batch size

            loss_epoch.append(loss.data.cpu().numpy())

    print(
        f"Validation Loss: Time (s): {time.time() - starttime:.1f} --- {loss_epoch[0] / total_step:.4f}")

    validation_loss = np.mean(loss_epoch)
    return validation_loss


def train(opt, encoder, decoder, logs, train_loader, test_loader, learning_rate, criterion, decay_rate):
    epoch_printer = EpochPrinter(train_loader, learning_rate, criterion)
    log_handler = LogHandler(opt, logs, train_loader,
                             criterion, encoder, learning_rate)

    decoder.to(device)
    decoder.train()

    optimizer = torch.optim.Adam(decoder.parameters(
    ), lr=learning_rate, weight_decay=1e-5)  # 1.5 * 10^-2 = 1.5/100

    training_losses = []
    validation_losses = []
    for epoch in range(opt["start_epoch"], opt["num_epochs"] + opt["start_epoch"]):

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=decay_rate)

        training_losses_epoch = []
        for step, (gt_audio_batch, _, _, _) in enumerate(train_loader):
            epoch_printer(step, epoch)

            # (batch_size, 1, 10240)
            gt_audio_batch = gt_audio_batch.to(device)

            encoding_per_module = encoder(gt_audio_batch)
            enc_audios = encoding_per_module[-1].to(device)

            # (batch_size, l, c)
            enc_audios = enc_audios.permute(0, 2, 1)  # (b, c, l)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            output_batch = decoder(enc_audios)
            loss = criterion(output_batch, gt_audio_batch) * \
                (1 / opt["batch_size_multiGPU"])

            # backward pass and optimization step
            loss.backward()
            optimizer.step()

            training_losses_epoch.append(loss.item())
            # </> end for step

        training_losses.append(np.mean(training_losses_epoch))
        validation_losses.append(validation_loss(
            encoder, decoder, test_loader, criterion))

        scheduler.step()
        print(f"LR: {scheduler.get_last_lr()}")

        log_handler(decoder, epoch, optimizer,
                    training_losses, validation_losses)

    # </> end epoch

    return decoder


def run_configuration(opt, experiment_name, GIM_MODEL_PATH, lr, decay_rate, criterion, decoder, num_epochs):
    torch.cuda.empty_cache()

    arg_parser.create_log_path(opt)
    opt['experiment'] = experiment_name
    opt['save_dir'] = f'{experiment_name}_experiment'
    opt['log_path'] = opt['log_path'] + "/DECODER"
    opt['log_path_latent'] = opt['log_path'] + "/latent_space"

    # opt['log_path'] = f'./logs/{experiment_name}_experiment'
    # opt['log_path_latent'] = f'./logs/{experiment_name}_experiment/latent_space'
    opt['num_epochs'] = num_epochs
    opt['batch_size'] = 171
    opt['batch_size_multiGPU'] = opt['batch_size']

    create_log_dir(opt['log_path'])

    logs = logger.Logger(opt)

    # load the data
    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt, dataset="de_boer_sounds_reshuffledv2", split_and_pad=False, train_noise=False, shuffle=True)

    encoder = GIM_Encoder(opt, path=GIM_MODEL_PATH)
    decoder = train(opt, encoder, decoder, logs, train_loader,
                    test_loader, lr, criterion, decay_rate)

    generate_predictions(opt, f"{experiment_name}_experiment", encoder,
                         criterion.name, lr, 1, decoder, model_nb=opt['num_epochs'] - 1)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    experiment_name = options_autoencoder["experiment_name"]
    GIM_MODEL_PATH = options_autoencoder["gim_model_path"]
    lr = options_autoencoder["lr"]
    decay_rate = options_autoencoder["decay_rate"]
    criterion = options_autoencoder["criterion"]
    decoder = options_autoencoder["decoder"]
    num_epochs = options_autoencoder["num_epochs"]

    run_configuration(opt, experiment_name, GIM_MODEL_PATH, lr,
                      decay_rate, criterion, decoder, num_epochs)
