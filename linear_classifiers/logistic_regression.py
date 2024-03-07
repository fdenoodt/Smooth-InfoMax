# example python call:
# python -m linear_classifiers.logistic_regression_syllables  final_bart/bart_full_audio_distribs_distr=true_kld=0 sim_audio_distr_false
# or
# python -m linear_classifiers.logistic_regression_syllables temp sim_audio_distr_true --overrides syllables_classifier_config.encoder_num=9

import torch
import torch.nn as nn
import time
import numpy as np

## own modules
from config_code.config_classes import OptionsConfig, ModelType, Dataset, ClassifierConfig
from models.full_model import FullModel
from options import get_options
from data import get_dataloader
from utils import logger
from arg_parser import arg_parser
from models import load_audio_model
from models.loss_supervised_syllables import Syllables_Loss

import wandb
import os

from utils.utils import retrieve_existing_wandb_run_id


def train(opt: OptionsConfig, context_model, loss: Syllables_Loss, logs: logger.Logger, train_loader, optimizer,
          wandb_is_on: bool):
    total_step = len(train_loader)
    print_idx = 100

    num_epochs = opt.syllables_classifier_config.num_epochs
    global_step = 0

    for epoch in range(num_epochs):
        loss_epoch = 0
        acc_epoch = 0

        if opt.model_type == ModelType.FULLY_SUPERVISED:
            context_model.train()
        else:
            context_model.eval()

        for i, (audio, _, label, _) in enumerate(train_loader):
            audio = audio.to(opt.device)
            label = label.to(opt.device)

            starttime = time.time()
            loss.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(opt.device)

            if opt.model_type == ModelType.FULLY_SUPERVISED:
                full_model: FullModel = context_model.module
                z = full_model.forward_through_all_modules(model_input)
            else:  # else: ModelType.ONLY_DOWNSTREAM_TASK
                with torch.no_grad():
                    full_model: FullModel = context_model.module
                    z = full_model.forward_through_all_modules(model_input)
                z = z.detach()

            # forward pass
            total_loss, accuracies = loss.get_loss(model_input, z, z, label)

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            sample_loss = total_loss.item()
            accuracy = accuracies.item()

            if wandb_is_on:
                label_type = "syllables" if opt.syllables_classifier_config.dataset.labels == "syllables" else "vowels"
                wandb.log({f"C {label_type}/Loss classification": sample_loss,
                           f"C {label_type}/Train accuracy": accuracy,
                           f"C {label_type}/Step": global_step})
                global_step += 1

            if i % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Accuracy: {:.4f}, Loss: {:.4f}".format(
                        epoch + 1,
                        num_epochs,
                        i,
                        total_step,
                        time.time() - starttime,
                        accuracy,
                        sample_loss,
                    )
                )
                starttime = time.time()

            loss_epoch += sample_loss
            acc_epoch += accuracy

        logs.append_train_loss([loss_epoch / total_step])


def test(opt, context_model, loss, data_loader, wandb_is_on: bool):
    loss.eval()
    accuracy = 0
    loss_epoch = 0

    with torch.no_grad():
        for i, (audio, _, label, _) in enumerate(data_loader):
            audio = audio.to(opt.device)
            label = label.to(opt.device)

            loss.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(opt.device)

            with torch.no_grad():
                full_model: FullModel = context_model.module
                z = full_model.forward_through_all_modules(model_input)

            z = z.detach()

            # forward pass
            total_loss, step_accuracy = loss.get_loss(model_input, z, z, label)

            accuracy += step_accuracy.item()
            loss_epoch += total_loss.item()

            if i % 10 == 0:
                print(
                    "Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}".format(
                        i, len(data_loader), loss_epoch / (i + 1), accuracy / (i + 1)
                    )
                )

    accuracy = accuracy / len(data_loader)
    loss_epoch = loss_epoch / len(data_loader)
    print("Final Testing Accuracy: ", accuracy)
    print("Final Testing Loss: ", loss_epoch)

    if wandb_is_on:
        label_type = "syllables" if opt.syllables_classifier_config.dataset.labels == "syllables" else "vowels"
        wandb.log({f"C {label_type}/FINAL Test accuracy": accuracy,
                   f"C {label_type}/FINAL Test loss": loss_epoch})
    return loss_epoch, accuracy


def main(syllables: bool, model_type: ModelType = ModelType.ONLY_DOWNSTREAM_TASK):
    opt: OptionsConfig = get_options()
    opt.model_type = model_type

    # fully supervised:
    # opt.model_type = ModelType.FULLY_SUPERVISED

    classifier_config: ClassifierConfig = opt.syllables_classifier_config
    classifier_config.dataset.labels = "syllables" if syllables else "vowels"

    assert opt.syllables_classifier_config is not None, "Classifier config is not set"
    assert opt.model_type in [ModelType.FULLY_SUPERVISED,
                              ModelType.ONLY_DOWNSTREAM_TASK], "Model type not supported"
    assert (opt.syllables_classifier_config.dataset.dataset in [Dataset.DE_BOER]), "Dataset not supported"

    # Check if the wandb_run_id.txt file exists
    wandb_is_on = False
    run_id, project_name = retrieve_existing_wandb_run_id(opt)
    # check if wandb alredy initialized
    if run_id is not None and not wandb_is_on:
        # Initialize a wandb run with the same run id
        wandb.init(id=run_id, resume="allow", project=project_name)
        wandb_is_on = True

    arg_parser.create_log_path(opt, add_path_var=f"linear_model_{classifier_config.dataset.labels}")

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    context_model, _ = load_audio_model.load_model_and_optimizer(
        opt,
        classifier_config,
        reload_model=True,  # if opt.model_type == ModelType.ONLY_DOWNSTREAM_TASK else False,
        calc_accuracy=True,
        num_GPU=1,
    )

    n_features = context_model.module.output_dim  # 256 or 512
    loss = Syllables_Loss(opt, n_features, calc_accuracy=True)
    learning_rate = opt.syllables_classifier_config.learning_rate

    if opt.model_type == ModelType.FULLY_SUPERVISED:
        context_model.train()
        params = list(context_model.parameters()) + list(loss.parameters())
    else:
        context_model.eval()
        params = list(loss.parameters())

    optimizer = torch.optim.Adam(params, lr=learning_rate)

    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt.syllables_classifier_config.dataset)

    logs = logger.Logger(opt)
    accuracy = 0

    try:
        # Train the model
        train(opt, context_model, loss, logs, train_loader, optimizer, wandb_is_on)

        # Test the model
        result_loss, accuracy = test(opt, context_model, loss, test_loader, wandb_is_on)

    except KeyboardInterrupt:
        print("Training interrupted, saving log files")

    logs.create_log(loss, accuracy=accuracy, final_test=True, final_loss=result_loss)

    print(f"Finished training {syllables}")

    return wandb, wandb_is_on


if __name__ == "__main__":
    pass
