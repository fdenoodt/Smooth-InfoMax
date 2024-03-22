# %%
import torch
from encoder.GIM_encoder import GIM_Encoder
from data import get_dataloader
from decoder.decoder_architectures import SimpleV2Decoder
from utils.helper_functions import create_log_dir
from options_interpolate import get_options
import soundfile as sf


def load_decoder(decoder_model_path, device):
    decoder = SimpleV2Decoder().to(device)
    decoder.load_state_dict(torch.load(
        decoder_model_path, map_location=device))
    return decoder


def set_dimension_equal_to(batch, b_idx, dim_idx, val):
    # batch is of shape: 171, 32, 2
    # so always change the last two points

    b, c, l = batch.shape
    for l_idx in range(l):
        batch[b_idx, dim_idx, l_idx] = val


def invent_latent_rnd(device):
    batch_enc = torch.rand((171, 32, 2)).to(device)

    return batch_enc


def invent_latent(device):
    # batch_enc = torch.randn((171, 32, 2)).to(device)
    batch_enc = torch.zeros((171, 32, 2)).to(device)

    # first sample: val(dim 0) := 1
    nb_dims = 32
    for dim in range(nb_dims):
        set_dimension_equal_to(batch_enc, b_idx=dim, dim_idx=dim, val=0.5)

    return batch_enc


if __name__ == "__main__":
    OPTIONS = get_options()
    DEVICE = OPTIONS["device"]

    CPC_MODEL_PATH = OPTIONS["cpc_model_path"]
    DECODER_MODEL_PATH = OPTIONS["decoder_model_path"]

    ENCODER = GIM_Encoder(OPTIONS, path=CPC_MODEL_PATH)
    ENCODER.encoder.eval()

    DECODER = load_decoder(DECODER_MODEL_PATH, DEVICE)
    DECODER.eval()

    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        OPTIONS, dataset="xxxx_sounds_reshuffledv2", split_and_pad=False, train_noise=False, shuffle=True)


    batch_enc_audio = invent_latent_rnd(DEVICE)

    batch_outp = DECODER(batch_enc_audio)

    target_dir = "invented_audios"
    sr = 16000
    create_log_dir(target_dir)
    for idx, outp in enumerate(batch_outp):
        outp_np = outp[0].cpu().detach().numpy()
        # Save .wav
        sf.write(f"{target_dir}/{idx}.wav", outp_np, sr)

        # Audio(outp[0].cpu().detach().numpy(), rate=16000)

    # # %%
    # for idx, (batch_org_audio, batch_filenames, _, _) in enumerate(test_loader):
    #     batch_org_audio = batch_org_audio.to(DEVICE)
    #     batch_per_module = ENCODER(batch_org_audio)
    #     batch_enc_audio = batch_per_module[-1].to(DEVICE)
    #     batch_enc_audio = batch_enc_audio.permute(0, 2, 1)  # (b, c, l)

    # print(batch_enc_audio.shape)
    # print(batch_outp.shape)

    # # %%

    # # play audio via IPython

    # inp = batch_org_audio[0][0].cpu().detach().numpy()
    # outp = batch_outp[0][0].cpu().detach().numpy()

    # # %%
    # Audio(outp, rate=16000)
    # Audio(inp, rate=16000)

    # # %%

    # print(batch_enc_audio.shape)
    # print(batch_enc_audio.min(), batch_enc_audio.max())

    # %%
    from __future__ import print_function
    from ipywidgets import interact
    import matplotlib.pyplot as plt
    import random

    def series(dots, colr):
        plt.figure(figsize=(10, 10))
        a, b = [], []
        for i in range(dots):
            a.append(random.randint(1, 100))
            b.append(random.randint(1, 100))
        plt.scatter(a, b, c=colr)
        return()
    interact(series, dots=(1, 100, 1), colr=["red", "orange", "brown"])

    None