# example python call:
# python -m post_hoc_analysis.interpretability.t_sne  final_bart/bart_full_audio_distribs_distr=true_kld=0 sim_audio_distr_false

# other example python call:
# python -m post_hoc_analysis.interpretability.t_sne temp sim_audio_distr_true --overrides syllables_classifier_config.encoder_num=0


import wandb

from arg_parser import arg_parser
## own modules
from config_code.config_classes import OptionsConfig, Dataset, DataSetConfig
from data import get_dataloader
from models import load_audio_model
from models.full_model import FullModel
from options import get_options
from utils.helper_functions import *
from utils.utils import retrieve_existing_wandb_run_id, set_seed
from interpretabil_util import plot_tsne_syllable, plot_histograms, scatter_3d

"""
This script is used to analyze the latent space of the audio encoder.
- t-SNE plots are created for the mean of the latent space and for the latent space itself
- histograms are created for the latent space
- 3D scatter plots are created for the latent space
"""


def _get_data_from_loader(loader, encoder: FullModel, opt: OptionsConfig, final_module: str):
    assert final_module in ["final", "final_cnn"]

    all_audio = np.array([])
    all_labels = np.array([])

    for i, (audio, _, label, _) in enumerate(loader):
        audio = audio.to(opt.device)

        with torch.no_grad():
            if final_module == "final":
                audio = encoder.forward_through_all_modules(audio)
            else:
                audio = encoder.forward_through_all_cnn_modules(audio)  # only cnn modules have kl divergence
            audio = audio.cpu().detach().numpy()  # (batch_size, seq_len, nb_channels)

            # vstack the audio
            if all_audio.size == 0:
                all_audio = audio
                all_labels = label
            else:
                all_audio = np.vstack((all_audio, audio))
                all_labels = np.hstack((all_labels, label))

    # If output from final_cnn, permute channels and seq_len
    if final_module == "final_cnn":
        all_audio = np.moveaxis(all_audio, 2, 1)

    return all_audio, all_labels


def main():
    opt: OptionsConfig = get_options()

    classifier_config = opt.syllables_classifier_config

    # Check if the wandb_run_id.txt file exists
    wandb_is_on = False
    run_id, project_name = retrieve_existing_wandb_run_id(opt)
    if run_id is not None:
        # Initialize a wandb run with the same run id
        wandb.init(id=run_id, resume="allow", project=project_name)
        wandb_is_on = True

    arg_parser.create_log_path(opt, add_path_var="post_hoc")

    # random seeds
    set_seed(opt.seed)

    load_existing_model = True
    context_model, _ = load_audio_model.load_model_and_optimizer(
        opt,
        classifier_config,
        reload_model=load_existing_model,
        calc_accuracy=True,
        num_GPU=1,
    )
    context_model.eval()
    logs = logger.Logger(opt)
    nb_channels = context_model.module.output_dim

    data_config = DataSetConfig(
        dataset=Dataset.DE_BOER,
        split_in_syllables=True,
        batch_size=128,
        limit_train_batches=1.0,
        limit_validation_batches=1.0,
        labels="syllables"
    )

    # retrieve data for classifier
    train_loader_syllables, _, test_loader_syllables, _ = get_dataloader.get_dataloader(data_config)
    all_audio, all_labels = _get_data_from_loader(train_loader_syllables, context_model.module, opt, "final")
    n = all_labels.shape[0]  # sqrt(1920) ~= 44

    # mean of seq len
    all_audio_mean = np.mean(all_audio, axis=1)  # (batch_size, nb_channels)
    lr, n_iter, perplexity = ('auto', 1000, int(float(np.sqrt(n))))
    plot_tsne_syllable(opt, all_audio_mean, all_labels, f"MEAN_SIM_{lr}_{n_iter}_{perplexity}",
                       lr=lr, n_iter=n_iter, perplexity=perplexity, wandb_is_on=wandb_is_on)

    data_config.labels = 'vowels'
    train_loader_syllables, _, test_loader_syllables, _ = get_dataloader.get_dataloader(data_config)
    all_audio, all_labels = _get_data_from_loader(train_loader_syllables, context_model.module, opt, "final_cnn")
    n = all_labels.shape[0]  # sqrt(1920) ~= 44

    _audio_per_channel = np.moveaxis(all_audio, 1, 0)
    scatter_3d(_audio_per_channel[0], _audio_per_channel[1], _audio_per_channel[2],
               all_labels, title=f"3D Latent Space of the First Three Dimensions", dir=opt.log_path,
               file=f"_ 3D latent space idices 0_1_2", show=False, wandb_is_on=wandb_is_on)
    #
    # retrieve full data that encoder was trained on
    data_config.split_in_syllables = False
    train_loader_full, _, test_loader, _ = get_dataloader.get_dataloader(data_config)
    all_audio, all_labels = _get_data_from_loader(train_loader_full, context_model.module, opt, "final_cnn")

    # plot histograms
    # (batch_size, seq_len, nb_channels) -> (nb_channels, batch_size, seq_len)
    audio_per_channel = np.moveaxis(all_audio, 2, 0)
    plot_histograms(opt, audio_per_channel, f"MEAN_SIM", max_dim=32, wandb_is_on=wandb_is_on)

    print("Finished")
    if wandb_is_on:
        wandb.finish()


if __name__ == "__main__":
    main()
    print("Finished")
