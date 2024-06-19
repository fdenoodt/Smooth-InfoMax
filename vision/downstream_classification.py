# Example usage:
# python -m vision.downstream_classification vis_dir vision_default

# Animals_with_Attributes dataset:
# python -m vision.downstream_classification vis_dir vision_default --overrides encoder_config.dataset.dataset=8 vision_classifier_config.dataset.dataset=8 encoder_config.num_epochs=200 vision_classifier_config.encoder_num=199

# STL dataset:
# python -m vision.downstream_classification vis_dir_stl vision_default --overrides encoder_config.num_epochs=200 vision_classifier_config.encoder_num=199
import torch
import numpy as np
import time

import os

from utils.utils import retrieve_existing_wandb_run_id
from vision.models.ClassificationModel import ClassificationModel

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from config_code.config_classes import OptionsConfig, ModelType, ClassifierConfig, Loss
from options import get_options
## own modules
from vision.data import get_dataloader
from vision.arg_parser import arg_parser
from vision.models import load_vision_model
from utils import logger, utils

import wandb


def train_logistic_regression(opt: OptionsConfig, context_model, classification_model, train_loader, wandb_is_on):
    total_step = len(train_loader)
    classification_model.train()

    starttime = time.time()
    global_step = 0

    for epoch in range(opt.vision_classifier_config.num_epochs):
        epoch_acc1 = 0
        epoch_acc5 = 0

        loss_epoch = 0
        for step, (img, target) in enumerate(train_loader):

            classification_model.zero_grad()

            model_input = img.to(opt.device)

            # TODO: IS THIS == 2 CORRECT?
            if opt.model_type == 2:  ## fully supervised training
                _, _, _, _, z = context_model(model_input)
            else:
                with torch.no_grad():
                    _, _, _, _, z, _ = context_model(model_input, target)
                z = z.detach()  # double security that no gradients go to representation learning part of model

            prediction = classification_model(z)

            target = target.to(opt.device)
            loss = criterion(prediction, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
            epoch_acc1 += acc1
            epoch_acc5 += acc5

            sample_loss = loss.item()
            loss_epoch += sample_loss

            if wandb_is_on and USE_WANDB:
                bias = opt.vision_classifier_config.bias
                deterministic_encoder = opt.encoder_config.deterministic
                wandb.log({f"C_bias={bias}_determistic_enc={deterministic_encoder}/Loss classification": sample_loss,
                           f"C_bias={bias}_determistic_enc={deterministic_encoder}/Train accuracy": acc1,
                           f"C_bias={bias}_determistic_enc={deterministic_encoder}/Train accuracy5": acc5,
                           f"C_bias={bias}_determistic_enc={deterministic_encoder}/Step": global_step,
                           f"C_bias={bias}_determistic_enc={deterministic_encoder}/Epoch": epoch})
                global_step += 1

            if step % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                        epoch + 1,
                        opt.vision_classifier_config.num_epochs,
                        step,
                        total_step,
                        time.time() - starttime,
                        acc1,
                        acc5,
                        sample_loss,
                    )
                )
                starttime = time.time()

        if opt.validate:
            # validate the model - in this case, test_loader loads validation data
            val_acc1, _, val_loss = test_logistic_regression(
                opt, context_model, classification_model, test_loader, wandb_is_on
            )

            if wandb_is_on and USE_WANDB:
                bias = opt.vision_classifier_config.bias
                deterministic_encoder = opt.encoder_config.deterministic
                wandb.log({f"C_bias={bias}_determistic_enc={deterministic_encoder}/Validation accuracy": val_acc1,
                           f"C_bias={bias}_determistic_enc={deterministic_encoder}/Validation loss": val_loss,
                           f"C_bias={bias}_determistic_enc={deterministic_encoder}/Epoch": epoch})

        print("Overall accuracy for this epoch: ", epoch_acc1 / total_step)


def test_logistic_regression(opt, context_model, classification_model, test_loader, wandb_is_on):
    total_step = len(test_loader)
    context_model.eval()
    classification_model.eval()

    starttime = time.time()

    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0

    for step, (img, target) in enumerate(test_loader):

        classification_model.zero_grad()

        model_input = img.to(opt.device)

        if opt.model_type == 2:  ## fully supervised training
            _, _, _, _, z = context_model(model_input)
        else:
            with torch.no_grad():
                _, _, _, _, z, _ = context_model(model_input, target)
            z = z.detach()  # double security that no gradients go to representation learning part of model

        prediction = classification_model(z)

        target = target.to(opt.device)
        loss = criterion(prediction, target)

        # calculate accuracy
        acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
        epoch_acc1 += acc1
        epoch_acc5 += acc5

        sample_loss = loss.item()
        loss_epoch += sample_loss

        if step % 10 == 0:
            print(
                "Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                    step, total_step, time.time() - starttime, acc1, acc5, sample_loss
                )
            )
            starttime = time.time()

    print("Testing Accuracy: ", epoch_acc1 / total_step)

    if wandb_is_on and USE_WANDB:
        bias = opt.vision_classifier_config.bias
        deterministic_encoder = opt.encoder_config.deterministic
        wandb.log({f"C_bias={bias}_determistic_enc={deterministic_encoder}/Test accuracy": epoch_acc1 / total_step,
                   f"C_bias={bias}_determistic_enc={deterministic_encoder}/Test accuracy5": epoch_acc5 / total_step,
                   f"C_bias={bias}_determistic_enc={deterministic_encoder}/Test loss": loss_epoch / total_step})

    return epoch_acc1 / total_step, epoch_acc5 / total_step, loss_epoch / total_step


if __name__ == "__main__":

    opt: OptionsConfig = get_options()
    USE_WANDB = opt.use_wandb
    TRAIN = opt.train

    opt.model_type = ModelType.ONLY_DOWNSTREAM_TASK

    opt.loss = Loss.SUPERVISED_VISUAL

    assert opt.vision_classifier_config is not None, "Classifier config is not set"

    wandb_is_on = False
    if USE_WANDB:
        wandb_is_on = False
        run_id, project_name = retrieve_existing_wandb_run_id(opt)
        if run_id is not None:
            # Initialize a wandb run with the same run id
            wandb.init(id=run_id, resume="allow", project=project_name)
            wandb_is_on = True

    dataset = opt.vision_classifier_config.dataset.dataset

    # order is important! first wandb.init, then create log path
    # Warning: doesnt consider module or layer, so is overwritten. (Saves storage as don't need to save them really)
    add_path_var = f"linear_model_vision_bias={opt.vision_classifier_config.bias}_deterministic_enc={opt.encoder_config.deterministic}"
    arg_parser.create_log_path(opt, add_path_var=add_path_var)

    # random seeds
    utils.set_seed(opt.seed)

    # load pretrained model
    context_model, _ = load_vision_model.load_model_and_optimizer(
        opt, reload_model=True, calc_loss=False, downstream_config=opt.vision_classifier_config
    )
    context_model.module.switch_calc_loss(False)

    # model_type=2 is supervised model which trains entire architecture; otherwise just extract features
    if opt.model_type != 2:
        context_model.eval()

    _, _, train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt.vision_classifier_config.dataset,
                                                                          purpose_is_unsupervised_learning=False)

    classification_model: ClassificationModel = load_vision_model.load_classification_model(opt)

    if opt.model_type == 2:
        params = list(context_model.parameters()) + list(classification_model.parameters())
    else:
        params = classification_model.parameters()

    optimizer = torch.optim.Adam(params)
    criterion = torch.nn.CrossEntropyLoss()

    logs = logger.Logger(opt)

    #### TRAINING ####
    if TRAIN:
        try:
            # Train the model
            train_logistic_regression(opt, context_model, classification_model, train_loader, wandb_is_on)

            # Test the model
            acc1, acc5, _ = test_logistic_regression(
                opt, context_model, classification_model, test_loader, wandb_is_on
            )

        except KeyboardInterrupt:
            print("Training got interrupted")

    logs.create_log(
        context_model,
        classification_model=classification_model,
        accuracy=acc1 if TRAIN else None,
        acc5=acc5 if TRAIN else None,
        final_test=True,
    )
    torch.save(
        context_model.state_dict(), os.path.join(opt.log_path, "context_model.ckpt")
    )

    #### SEND WEIGHTS TO WANDB (when bias=False) ####
    bias = opt.vision_classifier_config.bias
    if wandb_is_on and USE_WANDB and not bias:
        weights = list(classification_model.parameters())[0].detach().cpu().numpy()

        hidden_dim = opt.encoder_config.architecture.hidden_dim
        nb_classes = utils.get_nb_classes(dataset)
        assert weights.shape == (nb_classes, hidden_dim)

        weights = utils.rescale_between_neg1_and_1(weights)

        # Log weights as a table (3 rows, 256 columns)
        deterministic_encoder = opt.encoder_config.deterministic
        wandb.log({f"C_bias={bias}_determistic_enc={deterministic_encoder}/Vowel Classifier Weights tbl":
                       wandb.Table(data=weights, columns=[f"dim_{i}" for i in range(hidden_dim)])})

    if wandb_is_on and USE_WANDB:
        wandb.finish()
