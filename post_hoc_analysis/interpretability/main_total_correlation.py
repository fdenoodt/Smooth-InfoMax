import time
import numpy as np
import torch
import wandb
from sklearn.neighbors import KernelDensity
from arg_parser import arg_parser
from config_code.config_classes import OptionsConfig, Dataset, ClassifierConfig
from data import get_dataloader
from linear_classifiers.logistic_regression import get_z
from models import load_audio_model
from options import get_options
from utils.utils import set_seed, retrieve_existing_wandb_run_id


def generate_uncorrelated_data(batch_size, c, w, num_batches):
    latent_codes = None
    for _ in range(num_batches):
        z = torch.randn(batch_size, c, w, w)
        z = z.permute(0, 2, 3, 1).reshape(-1, c)  # (batch_size * w * w, c)
        if latent_codes is None:
            latent_codes = z
        else:
            latent_codes = torch.cat((latent_codes, z), dim=0)
    return latent_codes.cpu().numpy()


def generate_correlated_data(batch_size, c, w, num_batches):
    latent_codes = None
    for _ in range(num_batches):
        base = torch.randn(batch_size, 1, w, w)
        z = base.repeat(1, c, 1, 1) + 0.1 * torch.randn(batch_size, h, w, w)
        z = z.permute(0, 2, 3, 1).reshape(-1, c)
        if latent_codes is None:
            latent_codes = z
        else:
            latent_codes = torch.cat((latent_codes, z), dim=0)
    return latent_codes.cpu().numpy()


def _estimate_entropy(kde, samples):
    log_density = kde.score_samples(samples)
    return -np.mean(log_density)


def _calculate_total_correlation(latent_codes):
    """
    latent_codes: np.array of shape (num_samples, num_latent_codes)
    """
    assert len(latent_codes.shape) == 2, "Latent codes should be 2D array (num_samples, num_latent_codes)"

    marginal_kdes = []
    for i in range(latent_codes.shape[1]):
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(latent_codes[:, i].reshape(-1, 1))
        marginal_kdes.append(kde)

    joint_kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(latent_codes)

    marginal_entropies = []
    for i in range(latent_codes.shape[1]):
        entropy = _estimate_entropy(marginal_kdes[i], latent_codes[:, i].reshape(-1, 1))
        marginal_entropies.append(entropy)

    joint_entropy = _estimate_entropy(joint_kde, latent_codes)
    total_correlation = np.sum(marginal_entropies) - joint_entropy
    return total_correlation


def total_correlation(opt: OptionsConfig, context_model, data_loader, classifier: ClassifierConfig, nb_channel_indices,
                      nb_time_indices, max_nb_signals):
    print(f"Calculating total correlation for {max_nb_signals} signals, "
          f"with {nb_channel_indices} channels and {nb_time_indices} time indices")
    expected_channels = 512
    if nb_channel_indices > expected_channels:
        raise ValueError(f"Number of channels should be less than {expected_channels}")
    rnd_channel_indices = np.random.choice(expected_channels, nb_channel_indices, replace=False)

    # make data loader random
    data_loader.sampler.shuffle = True  # so can take first 1_000 signals randomly

    all_latent_codes = None
    for i, (audio, filename, _, audio_idx) in enumerate(data_loader):
        audio = audio.to(opt.device)
        model_input = audio.to(opt.device)
        z = get_z(opt, context_model, model_input,  # (batch_size, time, channels)
                  regression=classifier.bias,
                  which_module=classifier.encoder_module,
                  which_layer=classifier.encoder_layer)
        b_size, time, channels = z.shape

        z_subsampled = z[:, :, rnd_channel_indices]
        rnd_time_indices = np.random.choice(
            time,
            nb_time_indices,
            replace=False)  # take 5 random time indices for each signal
        z_subsampled = z_subsampled[:, rnd_time_indices, :]

        assert z.shape[2] == channels, "Latent code shape is not 512"
        if all_latent_codes is None:
            all_latent_codes = z_subsampled
        else:
            all_latent_codes = torch.cat((all_latent_codes, z_subsampled), dim=0)

        # Take first 1_000 signals
        if b_size * (i + 1) >= max_nb_signals:
            break

    print(f"Latent codes shape: {all_latent_codes.shape}")
    # (num_samples, time, channels) -> (num_samples*time, channels)
    all_latent_codes = all_latent_codes.reshape(-1, nb_channel_indices)
    all_latent_codes = all_latent_codes.cpu().detach().numpy()

    total_correlation_value = _calculate_total_correlation(all_latent_codes)
    print(f"Total correlation: {total_correlation_value}")
    return total_correlation_value

    # if opt.use_wandb:
    #     wandb_section = get_audio_libri_classific_key(
    #         "speakers",
    #         module_nb=classifier.encoder_module,
    #         layer_nb=classifier.encoder_layer,
    #         bias=classifier.bias,
    #         deterministic_encoder=opt.encoder_config.deterministic)
    #     wandb.log({f"{wandb_section}/Train Loss": sample_loss,
    #                f"{wandb_section}/Train Accuracy": accuracy})


def main():
    opt: OptionsConfig = get_options()
    classifier_config: ClassifierConfig = opt.speakers_classifier_config

    if opt.use_wandb:
        run_id, project_name = retrieve_existing_wandb_run_id(opt)
        wandb.init(id=run_id, resume="allow", project=project_name, entity=opt.wandb_entity)

    arg_parser.create_log_path(opt, add_path_var="total_correlation")

    assert opt.speakers_classifier_config.dataset.dataset in [
        Dataset.DE_BOER,
        Dataset.LIBRISPEECH,
        Dataset.LIBRISPEECH_SUBSET], "Dataset not supported"

    # random seeds
    set_seed(opt.seed)

    ## load model
    context_model, _ = load_audio_model.load_model_and_optimizer(
        opt,
        classifier_config,
        reload_model=True,
        calc_accuracy=True,
        num_GPU=1,
    )
    context_model.eval()

    # load dataset
    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt.speakers_classifier_config.dataset)


    channel_indices_list = [10, 50, 100, 512, 512]
    time_indices_list = [5, 5, 5, 5, 5]
    max_signals_list = [1_000, 1_000, 1_000, 1_000, np.inf]

    for nb_channel_indices, nb_time_indices, max_nb_signals in (
            zip(channel_indices_list, time_indices_list, max_signals_list)):
        print("*" * 50)
        print(
            f"nb_channel_indices: {nb_channel_indices}, "
            f"nb_time_indices: {nb_time_indices}, "
            f"max_nb_signals: {max_nb_signals}")
        print("*" * 50)
        for _ in range(1, 3):
            start = time.time()
            total_correlation(opt, context_model, test_loader, classifier_config,
                              nb_channel_indices=nb_channel_indices,
                              nb_time_indices=nb_time_indices,
                              max_nb_signals=max_nb_signals)
            print(f"Time: {time.time() - start} seconds")
            print("")

        print()
        print()
        print()

    if opt.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
    print("Done")
