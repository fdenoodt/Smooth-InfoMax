# copied from main_vowel_classifier_analysis.py as temporary solution to analyse libri speakers

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from arg_parser import arg_parser
from config_code.config_classes import OptionsConfig
from models import load_audio_model
from models.loss_supervised_speaker import Speaker_Loss
from options import get_options
from utils.utils import retrieve_existing_wandb_run_id, rescale_between_neg1_and_1, get_audio_libri_classific_key, \
    get_classif_log_path


def main():
    opt: OptionsConfig = get_options()
    bias = opt.speakers_classifier_config.bias

    assert not bias, "This script is only for the speaker classifier (bias=False)!!"

    if opt.use_wandb:
        run_id, project_name = retrieve_existing_wandb_run_id(opt)
        wandb.init(id=run_id, resume="allow", project=project_name, entity=opt.wandb_entity)

    # MUST HAPPEN AFTER wandb.init
    classifier_config = opt.speakers_classifier_config
    classif_module: int = classifier_config.encoder_module
    classif_layer: int = classifier_config.encoder_layer
    classif_path = get_classif_log_path(classifier_config, classif_module, classif_layer, bias, deterministic_encoder=opt.encoder_config.deterministic)
    arg_parser.create_log_path(
        opt, add_path_var=classif_path)
    context_model, _ = load_audio_model.load_model_and_optimizer(
        opt,
        opt.speakers_classifier_config,
        reload_model=True,
        calc_accuracy=True,
        num_GPU=1,
    )

    # the classifier is a part of the loss function
    n_labels = 251

    regr_hidden_dim = opt.encoder_config.architecture.modules[0].regressor_hidden_dim
    cnn_hidden_dim = opt.encoder_config.architecture.modules[0].cnn_hidden_dim
    if bias:
        n_features = regr_hidden_dim
    else:
        n_features = cnn_hidden_dim
    loss: Speaker_Loss = Speaker_Loss(
        opt, n_features, calc_accuracy=True, bias=bias
    )

    # Load the trained model
    model_path = opt.log_path + '/model_0.ckpt'
    loss.load_state_dict(torch.load(model_path))

    # Load a few data points
    context_model.eval()

    linear_classifier = loss.linear_classifier
    linear_classifier.eval()

    weights_and_biases = list(linear_classifier.parameters())
    assert len(weights_and_biases) == 1, f"The classifier also has a bias term, which is not supported here. len(temp)={len(weights_and_biases)}"
    weights = weights_and_biases[0].detach().cpu().numpy()

    # weights = list(linear_classifier.parameters())[0].detach().cpu().numpy()
    assert weights.shape == (n_labels, n_features)

    # axis=1 because we want to rescale each row (speaker) separately
    weights = rescale_between_neg1_and_1(weights, axis=1)
    weights = weights.T  # (dimensions, labels)

    # there seemed to be some problems with wandb so as a backup we save the weights as a numpy file as well
    np.save(f"{opt.log_path}/speaker_classifier_weights.npy", weights)
    # also as csv
    np.savetxt(f"{opt.log_path}/speaker_classifier_weights.csv", weights, delimiter=",")

    if opt.use_wandb:
        wandb_section = get_audio_libri_classific_key('speakers', module_nb=classif_module, layer_nb=classif_layer,
                                                      bias=bias, deterministic_encoder=opt.encoder_config.deterministic)
        # Log weights as a table (256 rows, 3 columns)
        wandb.log({f"{wandb_section}/Speaker Classifier Weights tbl":
                       wandb.Table(data=weights, columns=[f"label_{i}" for i in range(n_labels)])})

    if opt.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
