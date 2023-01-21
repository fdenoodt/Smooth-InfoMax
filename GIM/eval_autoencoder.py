# %%

import importlib
import decoder_architectures
import helper_functions
import torch.nn as nn
import IPython.display as ipd
from options import OPTIONS as opt
import torch
from utils import logger
from arg_parser import arg_parser
from data import get_dataloader
import matplotlib.pyplot as plt
import IPython.display as ipd
import random
from IPython.display import Audio

random.seed(0)

# %%

if(True):
    importlib.reload(decoder_architectures)
    importlib.reload(helper_functions)

    from decoder_architectures import *
    from helper_functions import *

# %%

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    decoder = OneLayerDecoder().to(device)
    decoder.load_state_dict(torch.load(
        "C:\\GitHub\\thesis-fabian-denoodt\\GIM\\logs\\RMSE_decoder_experiment\\model_0.ckpt", map_location=device))
    decoder.eval()

    experiment_name = 'RMSE_decoder'
    opt['experiment'] = experiment_name
    opt['save_dir'] = f'{experiment_name}_experiment'
    opt['log_path'] = f'./logs/{experiment_name}_experiment'
    opt['log_path_latent'] = f'./logs/{experiment_name}_experiment/latent_space'
    opt['num_epochs'] = 25
    opt['batch_size'] = 64

    logs = logger.Logger(opt)

    # load the data
    train_loader, _, test_loader, _ = get_dataloader.\
        get_de_boer_sounds_decoder_data_loaders(
            opt, layer_depth=1, GIM_encoder_path="./g_drive_model/model_180.ckpt")

    # %%

    (org_audio, enc_audio, _, _, _) = next(iter(train_loader))
    enc_audio = enc_audio.to(device)
    org_audio = org_audio.to(device)

    # %%
    outp = decoder(enc_audio)

    print(org_audio.shape)
    print(enc_audio.shape)
    print(outp.shape)

    # %%

    show_line_sequence(org_audio[0][0])
    show_line_sequence(outp[0][0])
    
    # %%
    

    # %%
    # decoder.eval()
    # train_loader, _, _, _ = get_dataloader.get_de_boer_sounds_decoder_data_loaders(opt, GIM_encoder=encoder_lambda)

    # enc_audios = None
    # org_audio = None
    # prediction = None
    # for step, (org_audio, enc_audio, filename, _, start_idx) in enumerate(train_loader):

    #     enc_audios = enc_audio.to(device)  # torch.Size([2, 1, 2047, 512])
    #     enc_audios = enc_audios.squeeze(dim=1)  # torch.Size([2, 2047, 512])
    #     enc_audios = enc_audios.permute(0, 2, 1)  # torch.Size([2, 512, 2047])

    #     org_audio = org_audio.to(device)  # torch.Size([2, 1, 10240])

    #     prediction = decoder(enc_audios)
    #     print(prediction.shape)

    #     break

    # %%
    Audio(prediction[0][0].to('cpu').detach().numpy(), rate=16000)
    Audio(org_audio[0][0].to('cpu').detach().numpy(), rate=16000)

    # %%
    # Audio(enc_audios[0][0].to('cpu').detach().numpy(), rate=3000)
    # Audio(enc_audios[0][1].to('cpu').detach().numpy(), rate=3000)
    # Audio(enc_audios[0][50].to('cpu').detach().numpy(), rate=3000)

    # %%
    # %%
    # multiple channels

    plot_spectrogram(enc_audios[0][0].to('cpu').detach().numpy(), "encoded")
    plot_spectrogram(org_audio[0][0].to('cpu').detach().numpy(), "original")

    # %%
    plt.plot(prediction[0][0].to('cpu').detach().numpy())
    plt.show()
    plt.plot(org_audio[0][0].to('cpu').detach().numpy())
    plt.show()
    plt.plot(enc_audios[0][100].to('cpu').detach().numpy())
    plt.show()

    # thought for later: its actually weird i was able to play enc as audio as enc is 512 x something
    # so huh? that means that a lot of info is already in first channel? what do other 511 channels then contain?
    decoder = train(decoder)

    # %%

    # %%
    """
    Observations:
    First layer decoded still contains the same sound, but with some added noise (could be because decoder hasn't trained very).
    However, the encoded first layer, still contains the exact sound as the original sound. It is however downsampled a lot -> from 16khz to ~3khz
    """
    # Audio(prediction[0][0].to('cpu').detach().numpy(), rate=16000)
    Audio(org_audio[0][0].to('cpu').detach().numpy(), rate=16000)

    # Audio(enc_audios[0][0].to('cpu').detach().numpy(), rate=3000)
    # Audio(enc_audios[0][1].to('cpu').detach().numpy(), rate=3000)
    # Audio(enc_audios[0][50].to('cpu').detach().numpy(), rate=3000)

    # %%
    Audio(enc_audios[0][1].to('cpu').detach().numpy(), rate=1000)

    # Audio(enc_audios[0][0].to('cpu').detach().numpy(), rate=16000)

    # print(enc_audios[0].shape)

    # multiple channels

    # %%

    # %%
    plot_spectrogram(enc_audios[0][0].to('cpu').detach().numpy(), "encoded")
    # %%
    # plot_spectrogram(prediction[0][0].to('cpu').detach().numpy(), "prediction")
    # %%
    plot_spectrogram(org_audio[0][0].to('cpu').detach().numpy(), "original")

    # %%
    # plt.plot(prediction[0][0].to('cpu').detach().numpy())
    # plt.show()

    plt.plot(org_audio[0][0].to('cpu').detach().numpy())
    plt.show()

    plt.plot(enc_audios[0][100].to('cpu').detach().numpy())
    plt.show()
    # %%

    # thought for later: its actually weird i was able to play enc as audio as enc is 512 x something
    # so huh? that means that a lot of info is already in first channel? what do other 511 channels then contain?

    # %%
