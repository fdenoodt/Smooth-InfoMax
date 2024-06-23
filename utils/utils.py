import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import torch

from config_code.config_classes import OptionsConfig, Dataset, DecoderConfig, ClassifierConfig
import wandb


def get_device(opt, input_tensor):
    if opt.device.type != "cpu":
        cur_device = input_tensor.get_device()
    else:
        cur_device = opt.device

    return cur_device


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Ensure no crash if # labels < maxk
        _, nb_classes = output.shape
        if nb_classes < maxk:
            maxk = nb_classes  # 100% accuracy
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # print(pred.min(), pred.max(), target.min(), target.max())

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct = correct.contiguous()  # required for pytorch V1.7 view()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def scatter(opt, x, colors, label):
    """
    creates scatter plot for t-SNE visualization
    :param x: 2-D latent space as output by t-SNE
    :param colors: labels for each datapoint in x, used to assign different colors to them
    :param idx: used for naming the file, to be able to track progress throughout training
    """
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
               c=palette[colors.ravel().astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis("off")
    ax.axis("tight")

    # save fig
    plt.savefig(
        os.path.join(opt.log_path_latent, f"latent_space_{label}.png"), dpi=120
    )

    # save data to numpy csv (x, colors)
    np.savetxt(os.path.join(opt.log_path_latent,
                            f"latent_space_x_{label}.csv"), x, delimiter=",")

    np.savetxt(os.path.join(
        opt.log_path_latent, f"latent_space_colors_{label}.csv"), colors, delimiter=",")

    plt.close()


def fit_TSNE_and_plot(opt, feature_space, speaker_labels, label):
    projection = TSNE(init='random',
                      learning_rate=200.0,
                      perplexity=30).fit_transform(feature_space)

    scatter(opt, projection, speaker_labels, label)


def retrieve_existing_wandb_run_id(opt: OptionsConfig):
    # Save the run id to a file in the logs directory
    if os.path.exists(os.path.join(opt.log_path, 'wandb_run_id.txt')):
        with open(os.path.join(opt.log_path, 'wandb_run_id.txt'), 'r') as f:
            text = f.read()
            # first line is the run id, second line is the project name (second line is optional)
            run_id = text.split('\n')[0]
            project_name = text.split('\n')[1] if len(text.split('\n')) > 1 else None

    # if file doesn't exist, return None
    else:
        run_id = None
        project_name = None

    assert run_id is not None, "Run id not found, set use_wandb to False in the config file to disable wandb logging"
    assert project_name is not None, "Project name not found, set use_wandb to False in the config file to disable wandb logging"

    return run_id, project_name


def set_seed(seed):
    if seed == -1:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def rescale_between_neg1_and_1(x, axis=0):
    """
    Rescale the input array to be between -1 and 1.
    e.g.: x = np.array([[1, 2, 3],
                        [4, 5, 6]])
    rescale_between_neg1_and_1(x, axis=0) -> array([[-1., -1., -1.],
    [ 1.,  1.,  1.]])"""

    # values are currently between ~ -1.5 and 1.5, so we rescale to -1 and 1
    return 2 * (x - x.min(axis=axis, keepdims=True)) / np.ptp(x, axis=axis, keepdims=True) - 1


def get_nb_classes(dataset: Dataset, args: None):
    # args only used in de_boer
    if dataset == Dataset.STL10:
        nb_classes = 10
    elif dataset == Dataset.ANIMAL_WITH_ATTRIBUTES:
        nb_classes = 50
    elif dataset in [Dataset.SHAPES_3D_SUBSET, Dataset.SHAPES_3D]:
        nb_classes = 4
    elif dataset == Dataset.DE_BOER and args == "vowels":
        nb_classes = 3
    elif dataset == Dataset.DE_BOER and args == "syllables":
        nb_classes = 9
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
    return nb_classes


def initialize_wandb(options: OptionsConfig, project_name, run_name):
    wandb.init(project=project_name, name=run_name, config=vars(options))
    # After initializing the wandb run, get the run id
    run_id = wandb.run.id
    # Save the run id to a file in the logs directory
    with open(os.path.join(options.log_path, 'wandb_run_id.txt'), 'w') as f:
        f.write(run_id)
        # write project name to file
        f.write(f"\n{project_name}")


def get_audio_classific_key(opt: OptionsConfig,
                            bias):  # used in logistic_regression.py and main_vowel_classifier_analysis.py
    """ONLY FOR DE_BOER DATASET. FOR LIBRI, USE get_audio_libri_classific_key() INSTEAD."""
    label_type = "syllables" if opt.syllables_classifier_config.dataset.labels == "syllables" else "vowels"
    module_nb = opt.syllables_classifier_config.encoder_module
    layer_nb = opt.syllables_classifier_config.encoder_layer
    return f"C bias={bias} {label_type} modul={module_nb} layer={layer_nb}"


def get_audio_libri_classific_key(label_type: str):
    assert label_type in ["phones", "speakers"], "Label type not supported"

    return f"libri_{label_type}_classifier"


def get_audio_decoder_key(decoder_config: DecoderConfig, loss_val):  # used in train_decoder.py, callbacks.py
    module_nb = decoder_config.encoder_module
    layer_nb = decoder_config.encoder_layer
    return f"Decoder_l={loss_val} modul={module_nb} layer={layer_nb}"


def get_classif_log_path(classifier_config, classif_module, classif_layer, bias):
    return f"linear_model_{classifier_config.dataset.labels}_modul={classif_module}_layer={classif_layer}_bias={bias}"
