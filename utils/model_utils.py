from typing import Optional

import torch
import torch.nn as nn
import os

from config_code.config_classes import OptionsConfig, ClassifierConfig, Dataset


def distribute_over_GPUs(opt: OptionsConfig, model, num_GPU):
    ## distribute over GPUs
    if opt.device.type != "cpu":
        if num_GPU is None:
            model = nn.DataParallel(model)
            num_GPU = torch.cuda.device_count()
            opt.encoder_config.dataset.batch_size_multiGPU = opt.encoder_config.dataset.batch_size * num_GPU
        else:
            assert (
                    num_GPU <= torch.cuda.device_count()
            ), "You cant use more GPUs than you have."
            model = nn.DataParallel(model, device_ids=list(range(num_GPU)))
            opt.encoder_config.dataset.batch_size_multiGPU = opt.encoder_config.dataset.batch_size * num_GPU
    else:
        model = nn.DataParallel(model)
        opt.encoder_config.dataset.batch_size_multiGPU = opt.encoder_config.dataset.batch_size

    model = model.to(opt.device)
    print("Let's use", num_GPU, "GPUs!")

    return model, num_GPU


def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def makeDeltaOrthogonal(weights, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = genOrthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    with torch.no_grad():
        weights[:, :, mid1, mid2] = q[: weights.size(0), : weights.size(1)]
        weights.mul_(gain)


def reload_weights_for_training_encoder(opt: OptionsConfig, model, optimizer, reload_model):
    return _reload_weights(True, opt, model, optimizer, reload_model, None)


def reload_weights_for_training_classifier(opt: OptionsConfig, model, optimizer, reload_model,
                                           classifier_config: ClassifierConfig):
    return _reload_weights(False, opt, model, optimizer, reload_model, classifier_config)


def _reload_weights(purpose_is_train_encoder: bool, opt: OptionsConfig, model, optimizer,
                    reload_model, classifier_config: Optional[ClassifierConfig]):
    ## reload weights for training of the linear classifier
    # old: if (opt.model_type == 0) and reload_model:  # or opt.model_type == 2)
    if not (purpose_is_train_encoder) and reload_model:  # or opt.model_type == 2)
        print("Loading weights from ", opt.model_path)

        if opt.experiment == "audio":
            model.load_state_dict(
                torch.load(
                    os.path.join(opt.model_path, f"model_{classifier_config.encoder_num}.ckpt"),
                    map_location=opt.device.type,
                )
            )
        else:
            for idx, layer in enumerate(model.module.encoder):
                model.module.encoder[idx].load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            f"model_{idx}_{classifier_config.encoder_num}.ckpt",
                        ),
                        map_location=opt.device.type,
                    )
                )

    ## reload weights and optimizers for continuing training
    elif opt.encoder_config.start_epoch > 0:
        print("Continuing training from epoch ", opt.encoder_config.start_epoch)

        if opt.experiment == "audio":
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        opt.model_path, "model_{}.ckpt".format(opt.encoder_config.start_epoch)
                    ),
                    map_location=opt.device.type,
                ),
                strict=False,
            )
        else:
            for idx, layer in enumerate(model.module.encoder):
                model.module.encoder[idx].load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            f"model_{idx}_{opt.encoder_config.start_epoch}.ckpt",
                        ),
                        map_location=opt.device.type,
                    )
                )

        optimizer.load_state_dict(
            torch.load(
                os.path.join(
                    opt.model_path,
                    "optim_{}.ckpt".format(opt.encoder_config.start_epoch),
                ),
                map_location=opt.device.type,
            )
        )
    else:
        print("Randomly initialized model")

    return model, optimizer


def modify_state_dict(nb_classes, state_dict, model, module_idx):
    # Remove unexpected keys
    for key in list(state_dict.keys()):
        if "W_k" in key:
            del state_dict[key]

    nb_hidden_dims = model.module.encoder[module_idx].loss.linear_classifier[0].weight.shape[1]
    # Add missing keys
    # TODO: 50 is the number of classes,
    state_dict["loss.linear_classifier.0.weight"] = torch.randn(nb_classes,
                                                                nb_hidden_dims)  # replace 50 and 256 with appropriate dimensions
    state_dict["loss.linear_classifier.0.bias"] = torch.randn(nb_classes)  # replace 50 with appropriate dimension

    return state_dict


def reload_weights_for_training_classifier_vision_experiment(opt: OptionsConfig, model, optimizer, reload_model,
                                                             classifier_config: ClassifierConfig):
    ## reload weights for training of the linear classifier
    if reload_model:
        print("Loading weights from ", opt.model_path)
        nb_classes = 10 if classifier_config.dataset.dataset == Dataset.STL10 else 50  # AWA2

        for idx, layer in enumerate(model.module.encoder):
            # Load the state dictionary
            state_dict = torch.load(
                os.path.join(
                    opt.model_path,
                    "model_{}_{}.ckpt".format(idx, classifier_config.encoder_num),
                ),
                map_location=opt.device.type,
            )

            # Modify the state dictionary
            state_dict = modify_state_dict(nb_classes, state_dict, model, idx)

            # Load the modified state dictionary into the model
            model.module.encoder[idx].load_state_dict(state_dict)

    else:
        print("Randomly initialized model")

    return model, optimizer


def reload_weights_for_training_encoder_vision_experiment(opt: OptionsConfig, model, optimizer, reload_model):
    ## reload weights for training of the linear classifier
    # model_type 0: train clas
    if reload_model:
        print("Continuing training from epoch ", opt.encoder_config.start_epoch)

        for idx, layer in enumerate(model.module.encoder):
            model.module.encoder[idx].load_state_dict(
                torch.load(
                    os.path.join(
                        opt.model_path,
                        "model_{}_{}.ckpt".format(idx, opt.encoder_config.start_epoch),
                    ),
                    map_location=opt.device.type,
                )
            )

        optimizer.load_state_dict(
            torch.load(
                os.path.join(
                    opt.model_path,
                    "optim_{}.ckpt".format(opt.encoder_config.start_epoch),
                ),
                map_location=opt.device.type,
            )
        )
    else:
        print("Randomly initialized model")

    return model, optimizer
