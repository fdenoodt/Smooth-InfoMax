# eg: for 512 dimensions
# SIM: full_pipeline_yyyyy/audio_FULL_PIPELINE_sim_xxxx_SIM=trueKLD=0.005 sim_audio_xxxx_distr_true
# GIM: full_pipeline_yyyyy/audio_FULL_PIPELINE_sim_xxxx_SIM=falseKLD=0 sim_audio_xxxx_distr_false

# now 32 dimensions (cnn) -> 16 dimensions (regressor)
# https://wandb.ai/eccv_tanmoy/SIM_ENCODER_FULL_PIPELINE_32dim/runs/1qy8znu4?workspace=user-oboii
# full_pipeline_yyyyy_32_16/audio_FULL_PIPELINE_sim_xxxx_32dim_SIM=trueKLD=0.001
# full_pipeline_yyyyy_32_16/audio_FULL_PIPELINE_sim_xxxx_32dim_SIM=falseKLD=0

import numpy as np
import torch

from config_code.config_classes import OptionsConfig
from data import get_dataloader
from decoder.lit_decoder import LitDecoder
from decoder.decoderr import Decoder
from models import load_audio_model
from utils.utils import set_seed
import wandb


def _get_data(opt: OptionsConfig, context_model: torch.nn.Module, decoder: Decoder):
    print("Loading data... SHUFFLE IS OFF!")
    _, _, test_loader, _ = get_dataloader.get_dataloader(opt.decoder_config.dataset, shuffle=False)

    with torch.no_grad():
        for batch in test_loader:
            (x, filename, label, _) = batch
            x = x.to(opt.device)
            full_model = context_model.module
            z = full_model.forward_through_all_cnn_modules(x)
            z = z.detach()
            x_reconstructed = decoder(z)
            break
    return x_reconstructed, x, z, filename


def _get_all_data(opt: OptionsConfig, context_model: torch.nn.Module, decoder: Decoder):
    print("Loading data... SHUFFLE IS OFF!")
    _, _, test_loader, _ = get_dataloader.get_dataloader(opt.decoder_config.dataset, shuffle=False)

    all_x_reconstructed = []
    all_x = []
    all_z = []
    all_filename = []

    with torch.no_grad():
        for batch in test_loader:
            (x, filename, label, _) = batch
            x = x.to(opt.device)
            full_model = context_model.module
            z = full_model.forward_through_all_cnn_modules(x)
            z = z.detach()
            x_reconstructed = decoder(z)

            all_x_reconstructed.append(x_reconstructed)
            all_x.append(x)
            all_z.append(z)
            all_filename.append(filename)

    return torch.cat(all_x_reconstructed), torch.cat(all_x), torch.cat(all_z), np.concatenate(all_filename)


def _reconstruct_audio(z: torch.Tensor, decoder: Decoder):
    x_reconstructed = decoder(z)
    return x_reconstructed


def _get_models(opt: OptionsConfig):
    context_model, _ = load_audio_model.load_model_and_optimizer(
        opt,
        opt.decoder_config,
        reload_model=True,
        calc_accuracy=False,
        num_GPU=1,
    )
    context_model.eval()
    context_model = context_model.to(opt.device)

    decoder: Decoder = load_audio_model.load_decoder(opt)
    decoder.eval()
    decoder = decoder.to(opt.device)
    return context_model, decoder


def _log_audio(key: str, audios: torch.Tensor, nb_files: int):
    audios = audios.squeeze(1).contiguous().cpu().data.numpy()
    audios = [audio_sample for audio_sample in audios[:nb_files]]

    wandb.log({key: [wandb.Audio(audio, sample_rate=16_000) for audio in audios]})


def _init_wandb(opt: OptionsConfig):
    wandb.init(project="DECODER_ANALYSIS",
               name=f"Dec L={opt.decoder_config.decoder_loss} conf={opt.config_file} "
                    f"sim={opt.encoder_config.architecture.modules[0].predict_distributions} "
                    f"{opt.encoder_config.architecture.modules[0].cnn_hidden_dim}x{opt.encoder_config.architecture.modules[0].regressor_hidden_dim}")
    for key, value in vars(opt).items():
        wandb.config[key] = value


def _interpolate(z1, z2, decoder, nb_interpolations):
    vals = np.linspace(0, 1, nb_interpolations)
    z_interpolated = torch.stack([z1 * val + z2 * (1 - val) for val in vals])
    x_interpolated = _reconstruct_audio(z_interpolated, decoder)
    return x_interpolated


def interpolate(z, filename, decoder, nb_interpolations=10):
    for x in range(20):
        z_sample_1 = z[x]
        z_sample_2 = z[x + 1]
        x_interpolated = _interpolate(z_sample_1, z_sample_2, decoder, nb_interpolations)
        _log_audio(f"{x} {filename[x]} to {filename[x + 1]}/interpolated", x_interpolated, nb_interpolations)


def zero_z(decoder, shape, device):
    z = torch.zeros(shape, device=device)
    x = _reconstruct_audio(z, decoder)
    _log_audio("zero_z", x, 10)


def mask_indices(z, decoder, indices, filenames):
    z_masked = z.clone()
    z_masked = z_masked.to(z.device)
    z_masked[:, indices, :] = 0
    x_masked = _reconstruct_audio(z_masked, decoder)
    for index in indices:
        _log_audio(f"masked/indices_{indices}/{filenames[index]}", x_masked[index], 1)

    return x_masked


def mask_all_but_indices(z, decoder, indices, filenames):
    z_masked = z.clone()
    z_masked = z_masked.to(z.device)
    z_masked[:, indices, :] = 0
    z_masked = z_masked * 0.0
    z_masked[:, indices, :] = z[:, indices, :]
    x_masked = _reconstruct_audio(z_masked, decoder)
    for index in indices:
        _log_audio(f"preserved/indices_{indices}/{filenames[index]}", x_masked[index], 1)

    return x_masked


def log_z_vals(z, filenames, device):
    # log first 10 z's, and their respective filenames as a table
    z = z.to(device)
    for i in range(10):
        wandb.log({f"z/{filenames[i]}/{i}":
                       wandb.Table(data=[z[i].cpu().numpy()],
                                   columns=[f"dim_{i}" for i in range(z.shape[1])])})
        # columns=[f"dim_{i}" for i in _w[:32]]


def main():
    from options import get_options

    # Load options
    opt = get_options()
    set_seed(opt.seed)
    _init_wandb(opt)

    context_model, decoder = _get_models(opt)

    # Load the data
    x_reconstructed, x, z, filename = _get_data(opt, context_model, decoder)
    batch_size, dims, nb_frames = z.shape
    np.save("z.npy", z.cpu().numpy())
    np.save("filename.npy", filename)

    x_randn = _reconstruct_audio(torch.randn_like(z), decoder)
    x_zeros = _reconstruct_audio(torch.zeros_like(z), decoder)

    interpolate(z, filename, decoder)
    zero_z(decoder, z.shape, opt.device)

    # 10 rnd indices between 0 and dims-1
    rnd_indices = np.random.choice(dims, 10, replace=False)
    mask_indices(z, decoder, rnd_indices, filename)
    mask_all_but_indices(z, decoder, rnd_indices, filename)

    important_indices = np.array([4, 9, 10, 13, 15, 21, 23, 25, 33]) - 2
    mask_indices(z, decoder, important_indices, filename)
    mask_all_but_indices(z, decoder, important_indices, filename)

    # log_z_vals(z, filename, opt.device)

    # save z to numpy file

    # take z_sample_1 but apply mask on irrelevant dimensions (eg. 0s on all but dim 0, 4 and 48)
    # z_masked = z_sample_1.clone()  # shape: (z_dim, nb_frames)
    # z_masked[0, :] = 0 # z_masked[4, :] = 0 # z_masked[48, :] = 0
    # # 10 rnd indices
    # indices = np.random.choice(z_masked.shape[0], 10, replace=False)
    # z_masked[indices, :] = 0
    # x_masked = _reconstruct_audio(z_masked.unsqueeze(0), decoder)

    # # Mask the important indices
    # important_indices = [441, 186, 413, 85, 488, 424, 327, 433, 24, 245, 99, 231, 218, 454, 387, 308, 0, 266, 351, 138,
    #                      163, 491, 495, 195, 500, 18, 46, 249, 236, 244, 127, 230]
    # z_masked = z_sample_1.clone()
    # z_masked[important_indices, :] = 0
    # x_masked = _reconstruct_audio(z_masked.unsqueeze(0), decoder)
    #
    # # Now mask all BUT the important indices
    # z_masked = z_sample_1.clone()
    # z_masked[important_indices, :] = 0
    # z_masked = z_masked * 0.0
    # z_masked[important_indices, :] = z_sample_1[important_indices, :]
    # x_masked_reverse = _reconstruct_audio(z_masked.unsqueeze(0), decoder)
    #
    # # mask each dimension at a time
    # z_masked = z_sample_1.clone()
    # # repeat 512 times
    # z_masked = z_masked.repeat(512, 1, 1).reshape(dims, dims, nb_frames)
    # for i in range(dims):
    #     z_masked[i, :i, :] = 0
    # x_masked_single_dim = _reconstruct_audio(z_masked, decoder)
    #
    # # remove half of the dimensions
    # z_masked = z_sample_1.clone()
    # z_masked[dims//2:, :] = 0
    # x_masked_half = _reconstruct_audio(z_masked.unsqueeze(0), decoder)
    #
    # # remove other half of the dimensions
    # z_masked = z_sample_1.clone()
    # z_masked[:dims//2, :] = 0
    # x_other_half = _reconstruct_audio(z_masked.unsqueeze(0), decoder)
    #
    # # remove 75% of the dimensions
    # z_masked = z_sample_1.clone()
    # z_masked[dims//4:, :] = 0
    # x_masked_75 = _reconstruct_audio(z_masked.unsqueeze(0), decoder)
    #
    # # remove 25% of the dimensions
    # z_masked = z_sample_1.clone()
    # z_masked[:3*dims//4, :] = 0
    # x_masked_25 = _reconstruct_audio(z_masked.unsqueeze(0), decoder)

    _log_audio("reconstr", x_reconstructed, 10)
    _log_audio("gt", x, 10)

    _log_audio("std_normal", x_randn, 10)
    _log_audio("zeros", x_zeros, 10)

    # _log_audio("masked", x_masked, 10)
    # _log_audio("masked_reverse", x_masked_reverse, 10)
    # _log_audio("masked_single_dim", x_masked_single_dim, 100)
    #
    # _log_audio("masked_half", x_masked_half, 10)
    # _log_audio("other_half", x_other_half, 10)
    # _log_audio("masked_75", x_masked_75, 10)
    # _log_audio("masked_25", x_masked_25, 10)

    wandb.finish()

    return x_reconstructed, x


if __name__ == "__main__":
    x_reconstructed, x = main()
