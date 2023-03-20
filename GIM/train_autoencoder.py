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
        enc_audio = enc_audio.permute(0, 2, 1) # (b, c, l)


        with torch.no_grad():
            outputs = model(enc_audio)
            loss = criterion(outputs, org_audio)
            loss = torch.mean(loss, 0)

            loss_epoch.append(loss.data.cpu().numpy())

    print(
        f"Validation Loss: Time (s): {time.time() - starttime:.1f} --- {loss_epoch[0] / total_step:.4f}")

    validation_loss = np.mean(loss_epoch)
    return validation_loss


def train(decoder, logs, train_loader, test_loader, learning_rate, criterion):
    epoch_printer = EpochPrinter(train_loader, learning_rate, criterion)
    log_handler = LogHandler(opt, logs, train_loader,
                             criterion, encoder, learning_rate)

    normalize_func = compute_normalizer(train_loader, encoder)

    decoder.to(device)
    decoder.train()

    optimizer = torch.optim.Adam(decoder.parameters(
    ), lr=learning_rate, weight_decay=1e-5)  # 1.5 * 10^-2 = 1.5/100

    training_losses = []
    validation_losses = []
    for epoch in range(opt["start_epoch"], opt["num_epochs"] + opt["start_epoch"]):

        training_losses_epoch = []
        for step, (gt_audio_batch, _, _, _) in enumerate(train_loader):
            epoch_printer(step, epoch)

            # (batch_size, 1, 10240)
            gt_audio_batch = gt_audio_batch.to(device)

            encoding_per_module = encoder(gt_audio_batch)
            
            # (batch_size, l, c)
            enc_audios = normalize_func(encoding_per_module[-1].to(device))
            enc_audios = enc_audios.permute(0, 2, 1) # (b, c, l)

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

        log_handler(decoder, epoch, optimizer,
                    training_losses, validation_losses)

    # </> end epoch

    return decoder


if __name__ == "__main__":
    torch.cuda.empty_cache()

    arg_parser.create_log_path(opt)

    experiment_name = 'GIM_DECODER'
    opt['experiment'] = experiment_name
    opt['save_dir'] = f'{experiment_name}_experiment'
    opt['log_path'] = f'./logs/{experiment_name}_experiment'
    opt['log_path_latent'] = f'./logs/{experiment_name}_experiment/latent_space'
    opt['num_epochs'] = 20  # 30
    opt['batch_size'] = 64 + 32
    opt['batch_size_multiGPU'] = opt['batch_size']

    # "DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/model_360.ckpt"
    GIM_MODEL_PATH = r"D:\thesis_logs\logs\audio_noise=F_dim=32_distr_wo_nonlin_kld_weight=0.032 !!/model_799.ckpt"

    create_log_dir(opt['log_path'])

    logs = logger.Logger(opt)

    # load the data
    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt, dataset="de_boer_sounds", split_and_pad=False, train_noise=False, shuffle=True)

    criterion = MEL_LOSS()
    lr = 1e-3
    encoder = GIM_Encoder(opt, path=GIM_MODEL_PATH)
    decoder = SimpleV1Decoder()
    decoder = train(decoder, logs, train_loader,
                    test_loader, lr, criterion)

    generate_predictions(encoder, criterion.name, lr, 1, decoder, model_nb=opt['num_epochs'] - 1)

    # criterion = MSE_Loss()
    # criterion = FFTLoss()
    # for criterion in [MSE_Loss(), MSE_AND_SPECTRAL_LOSS(128), MSE_AND_SPECTRAL_LOSS(8192)]:
        # for lr in [1e-3, 1e-2, 1e-4, 1e-5]:
    # for lr in [1e-3, 1e-4, 1e-5]:
    #     # for layer_depth, Decoder in zip([1, 2, 3, 4], [GimL1Decoder, GimL2Decoder, GimL3Decoder, GimL4Decoder]):
    #     for layer_depth, Decoder in zip([4], [GimL4Decoder]):
    #         encoder = GIM_Encoder(opt, layer_depth=layer_depth, path="DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/model_360.ckpt")
    #         decoder = Decoder()
    #         decoder = train(decoder, logs, train_loader, test_loader, lr, criterion)

    # SOLEY FFT
    # criterion = FFTLoss(fft_size=10240)
    # for lr in [1e-3, 1e-2, 1e-5, 1e-4]:
    #     for layer_depth, Decoder in zip([4], [GimL4Decoder]):
    #         encoder = GIM_Encoder(opt, layer_depth=layer_depth, path="DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/model_360.ckpt")
    #         decoder = Decoder()
    #         decoder = train(decoder, logs, train_loader, test_loader, lr, criterion)
    #         generate_predictions(encoder, criterion.name, lr, layer_depth, decoder, model_nb=opt['num_epochs'] - 1)

    # mse + sft
    # for lambd in [10, 1, 0.1, 0.01]:
    #     bin_size = 8192
    #     criterion = MSE_AND_FFT_LOSS(fft_size=10240, lambd=lambd)
    #     for lr in [1e-5, 1e-4, 1e-3, 1e-2]:
    #         for layer_depth, Decoder in zip([4], [GimL4Decoder]):
    #             encoder = GIM_Encoder(opt, path="DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/model_360.ckpt")
    #             decoder = Decoder()
    #             decoder = train(decoder, logs, train_loader, test_loader, lr, criterion)

    #             generate_predictions(encoder, criterion.name, lr, layer_depth, decoder, model_nb=opt['num_epochs'] - 1)

    # for lambd in [10, 1, 0.1, 0.01]:
    #     criterion = MSE_AND_SPECTRAL_LOSS(bin_size, lambd)
    #     # for criterion in [MSE_Loss(), MSE_AND_SPECTRAL_LOSS(128), MSE_AND_SPECTRAL_LOSS(8192)]:
    #     for lr in [1e-5, 1e-4, 1e-3, 1e-2]:
    #         # for layer_depth, Decoder in zip([1, 2, 3, 4], [GimL1Decoder, GimL2Decoder, GimL3Decoder, GimL4Decoder]):
    #         for layer_depth, Decoder in zip([4], [GimL4Decoder]):
    #             encoder = GIM_Encoder(opt, layer_depth=layer_depth, path="DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/model_360.ckpt")
    #             decoder = Decoder()
    #             decoder = train(decoder, logs, train_loader, test_loader, lr, criterion)

    #             generate_predictions(encoder, criterion.name, lr, layer_depth, decoder, model_nb=opt['num_epochs'] - 1)

    torch.cuda.empty_cache()

    # %%
