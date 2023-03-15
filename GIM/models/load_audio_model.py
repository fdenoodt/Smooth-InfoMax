import torch

from models import full_model
from utils import model_utils


def load_model_and_optimizer(
    opt, lr, reload_model=False, calc_accuracy=False, num_GPU=None
):
    architecture = opt["architecture"]
    kernel_sizes = architecture["kernel_sizes"]
    strides = architecture["strides"]
    padding = architecture["padding"]

    cnn_hidden = architecture["cnn_hidden_dim"]
    regressor_hidden = architecture["regressor_hidden_dim"]

    if opt["model_splits"] > 1:
        assert len(kernel_sizes) == len(strides) == len(padding) == opt["model_splits"], (
            "Inconsistent size of network parameters (kernels, strides and padding)"
        )

    # Initialize model.
    model = full_model.FullModel(
        opt,
        kernel_sizes=kernel_sizes,
        strides=strides,
        padding=padding,
        cnn_hidden_dim=cnn_hidden,
        regressor_hidden_dim=regressor_hidden,
        calc_accuracy=calc_accuracy,
    )

    # Run on only one GPU for supervised losses.
    if opt["loss"] == 2 or opt["loss"] == 1:
        num_GPU = 1

    model, num_GPU = model_utils.distribute_over_GPUs(
        opt, model, num_GPU=num_GPU)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model, optimizer = model_utils.reload_weights(
        opt, model, optimizer, reload_model)

    model.train()
    print(model)

    return model, optimizer
