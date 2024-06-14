from decoder.eval_decoder import _get_models, _reconstruct_audio, _get_data, _get_all_data

# Import necessary modules
import numpy as np
import torch
import IPython.display as ipd


# hacky way to get certain variables in the global scope

# nb_dims = 512  # Constant. In theory this can be changed, but only the 512-dim architecture is provided


def create_get_options_func(type, nb_dims=512):
    model = f"{type}_{nb_dims}"
    if model == "SIM_512":
        from configs.sim_audio_xxxx_distr_true import _get_options
        experiment_name = "SIM=trueKLD=0.01"
    elif model == "GIM_512":
        from configs.sim_audio_xxxx_distr_false import _get_options
        experiment_name = "SIM=falseKLD=0"
    else:
        raise ValueError(f"model: {model} not found")

    get_options = lambda: _get_options(experiment_name)
    return get_options


def plot(z, opt, decoder, savename=""):
    z_tensor = torch.from_numpy(z).float().to(opt.device)
    x_reconstructed = _reconstruct_audio(z_tensor, decoder)
    audio = x_reconstructed.squeeze().cpu().detach().numpy()

    # display audio time series
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 2))

    plt.plot(audio[:-10])
    # set x axis between 0 and 1000
    plt.ylim(-0.35, 0.35)

    if savename:
        plt.title(savename)
        plt.savefig(f"temp/pdf/audio_{savename}.pdf")
        plt.savefig(f"temp/png/audio_{savename}.png")
        # tikz
        try:
            import tikzplotlib
            tikzplotlib.save(f"temp/tex/audio_{savename}.tex")
        except ImportError:
            pass

    plt.show()


def plot_reduced_resolution(z, opt, decoder, savename="", downsample_factor=10):
    z_tensor = torch.from_numpy(z).float().to(opt.device)
    x_reconstructed = _reconstruct_audio(z_tensor, decoder)
    audio = x_reconstructed.squeeze().cpu().detach().numpy()

    # Downsample the audio data
    audio_downsampled = audio[::downsample_factor]

    # Create a new x-axis array that corresponds to the downsampled audio data
    # This time it's in seconds
    x = np.linspace(0, len(audio) / 16000, len(audio_downsampled))

    # Display audio time series
    import matplotlib.pyplot as plt

    plt.plot(x, audio_downsampled)
    plt.ylim(-0.35, 0.35)

    # x, y labels
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    if savename:
        plt.title(savename)
        plt.savefig(f"temp/pdf/audio_{savename}.pdf")
        plt.savefig(f"temp/png/audio_{savename}.png")
        # tikz
        try:
            import tikzplotlib
            tikzplotlib.save(f"temp/tex/audio_{savename}.tex")
        except ImportError:
            pass

    plt.show()


def decode_audio(z, opt, decoder):
    z_tensor = torch.from_numpy(z).float().to(opt.device)
    x_reconstructed = _reconstruct_audio(z_tensor, decoder)
    audio = x_reconstructed.squeeze().cpu().detach().numpy()
    return audio


def decode_and_listen(z, opt, decoder):
    z_tensor = torch.from_numpy(z).float().to(opt.device)
    x_reconstructed = _reconstruct_audio(z_tensor, decoder)
    audio = x_reconstructed.squeeze().cpu().detach().numpy()
    # remove last 10 samples to avoid clicking sound
    audio = audio[:-10]

    from IPython.display import display
    display(ipd.Audio(audio, rate=16_000))


def get_two_zs(z, filenames, idx1, idx2, print_names: bool = True, nb_dims=512) -> (np.ndarray, np.ndarray, str, str):
    assert idx1 != idx2

    z = z.cpu().numpy()
    z1 = z[idx1].reshape(1, nb_dims, -1)  # (1, 32, 64)
    z2 = z[idx2].reshape(1, nb_dims, -1)  # (1, 32, 64)
    z1_file = filenames[idx1]
    z2_file = filenames[idx2]
    if print_names:
        print(z1_file, z2_file)

    if idx1 in [27, 15] and idx2 in [27, 15]:
        assert z1_file in ["bibibi_1", "bagaga_1"]
        assert z2_file in ["bibibi_1", "bagaga_1"]

    return z1, z2, z1_file, z2_file


def print_z_names(filenames):
    for i, f in enumerate(filenames):
        print(f"Idx: {i}: {f}")


def interpolate(z1, z2, nb_interpolations=10):
    vals = np.linspace(0, 1, nb_interpolations)
    z_interpolated = np.stack([z1 * val + z2 * (1 - val) for val in vals])
    return z_interpolated


def _interpolate_partial(z1, z2, indices, nb_interpolations=10, nb_dims=512):
    z_interpolated = np.zeros((nb_interpolations, nb_dims, 64))
    for i in range(nb_interpolations):
        val = i / (nb_interpolations - 1)
        z_interpolated[i] = z1.copy()
        z_interpolated[i, indices] = z1[0, indices] * val + z2[0, indices] * (1 - val)
    return z_interpolated


def interpolate_partial(z1, z2, nb_most_important_dims, indices, nb_interpolations=10, nb_dims=512):
    assert nb_most_important_dims <= nb_dims and nb_most_important_dims > 0, f"nb_most_important_dims should be between 0 and {nb_dims}"

    z_1_single_timeframe = z1.mean(axis=2)  # (1, 32)
    z_2_single_timeframe = z2.mean(axis=2)  # (1, 32)

    mse = (z_1_single_timeframe - z_2_single_timeframe) ** 2
    # Get the most important dimensions
    dims = np.argsort(mse.squeeze())[::-1][:nb_most_important_dims]  # in descending order
    return _interpolate_partial(z1, z2, dims, nb_interpolations, nb_dims)


def bar_chart(z, savename="", nb_dims=512):
    import matplotlib.pyplot as plt
    # plot window size
    plt.figure(figsize=(26, 5))

    plt.bar(range(nb_dims), z.squeeze())

    # if no negative values:
    if np.all(z >= 0):
        pass
    else:
        # y axis between -2 and 2
        plt.ylim(-4, 4)

    plt.xlim(0, nb_dims)

    # set the resolution of the x axis
    # plt.xticks(range(nb_dims))

    if savename:
        plt.savefig(f"temp/pdf/bar_{savename}.pdf")
        plt.savefig(f"temp/png/bar_{savename}.png")
        # tikz
        try:
            import tikzplotlib
            tikzplotlib.save(f"temp/tex/bar_{savename}.tex")
        except ImportError:
            pass

    plt.show()


def setup_demo(type, decoder):
    get_options = create_get_options_func(type)
    opt = get_options()
    from config_code.config_classes import DecoderLoss

    if decoder == "MSE":
        opt.decoder_config.decoder_loss = DecoderLoss.MSE
    elif decoder == "MSE+MEL":
        opt.decoder_config.decoder_loss = DecoderLoss.MSE_MEL
    else:
        raise ValueError(f"Only 'MSE' and 'MSE+MEL' are supported for decoder. Got: {decoder}")

    context_model, decoder = _get_models(opt)
    _, _, z_data, filenames = _get_data(opt, context_model, decoder)
    return z_data, filenames, opt, context_model, decoder
