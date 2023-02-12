# %%
import importlib
from GIM_encoder import GIM_Encoder
import decoder_architectures
import helper_functions
from options import OPTIONS as opt
import torch
from utils import logger
from data import get_dataloader
import matplotlib.pyplot as plt
import IPython.display as ipd
import random
from options import OPTIONS as opt
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

    decoder = GimL3Decoder().to(device)
    # decoder = OneLayerDecoder().to(device)
    # model_path = "./logs/RMSE_decoder_GIM_layer3_spectral_loss_experiment/model_19.pt"
    model_path = "./logs/RMSE_decoder_GIM_layer3_MSE_loss_experiment/model_49.pt"
    # model_path = "./logs/RMSE_decoder_GIM_layer3_MSE_SPECTRAL_loss_experiment/model_49.pt"
    decoder.load_state_dict(torch.load(model_path, map_location=device))
    decoder.eval()

    logs = logger.Logger(opt)

    experiment_name = 'RMSE_decoder_GIM_layer3'
    opt['experiment'] = experiment_name
    opt['save_dir'] = f'{experiment_name}_experiment'
    opt['log_path'] = f'./logs/{experiment_name}_experiment'
    opt['log_path_latent'] = f'./logs/{experiment_name}_experiment/latent_space'
    opt['batch_size'] = 64

    # load the data
    train_loader, _, test_loader, _ = get_dataloader.\
        get_de_boer_sounds_decoder_data_loaders(opt)

    encoder = GIM_Encoder(opt, layer_depth=3, path="DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/model_360.ckpt")

    # %%
    normalize_func = compute_normalizer(train_loader, encoder)

    # %%

    (org_audio, _, _, _) = next(iter(train_loader))
    org_audio = org_audio.to(device)
    enc_audio = normalize_func(encoder(org_audio).to(device))

    # %%

    # for idx, module in enumerate(decoder.modules()):
    #     if (idx != 0):
    #         print(idx)
    #         print(module.weight)
    #         print(module.bias)

    # %%
    outp = decoder(enc_audio)


    print(org_audio.shape)
    print(enc_audio.shape)
    print(outp.shape)

    # %%

    show_line_sequence(org_audio[0][0])
    show_line_sequence(outp[0][0])


    def det_np(data):
        #detach + numpy
        return data.to('cpu').detach().numpy()
    # %%

    Audio(det_np(outp[0][0]), rate=16000)
    
    # %%
    Audio(det_np(org_audio[0][0]), rate=16000)

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
