# example usage:
# python -m vision.main_vision vis_dir vision_default
# or for Animals_with_Attributes:
# python -m vision.main_vision vis_dir vision_default --overrides encoder_config.dataset.dataset=8 vision_classifier_config.dataset.dataset=8 encoder_config.num_epochs=200

import torch
import time
import numpy as np
import os

from config_code.config_classes import ModelType, OptionsConfig
#### own modules
from utils import logger
from vision.arg_parser import arg_parser
from vision.models import load_vision_model
from vision.data import get_dataloader
from options import get_options

import wandb


def validate(opt: OptionsConfig, model, test_loader):
    total_step = len(test_loader)
    model_splits = 3  # TODO
    # model_splits = opt.model_splits

    loss_epoch = [0 for i in range(model_splits)]
    starttime = time.time()

    for step, (img, label) in enumerate(test_loader):
        model_input = img.to(opt.device)
        label = label.to(opt.device)

        loss, _, _, _ = model(model_input, label, n=opt.train_module)
        loss = torch.mean(loss, 0)

        loss_epoch += loss.data.cpu().numpy()

    for i in range(opt.model_splits):
        print(
            "Validation Loss Model {}: Time (s): {:.1f} --- {:.4f}".format(
                i, time.time() - starttime, loss_epoch[i] / total_step
            )
        )

    validation_loss = [x / total_step for x in loss_epoch]
    return validation_loss


def train(opt: OptionsConfig, model: torch.nn.Module):
    total_step = len(train_loader)
    model.module.switch_calc_loss(True)

    print_idx = 100

    starttime = time.time()
    # cur_train_module = opt.train_module # TODO
    cur_train_module = 3

    global_step = 0

    for epoch in range(opt.encoder_config.start_epoch, opt.encoder_config.num_epochs + opt.encoder_config.start_epoch):

        # model_splits = opt.model_splits # TODO
        model_splits = 3
        loss_epoch = [0 for _ in range(model_splits)]
        loss_updates = [1 for _ in range(model_splits)]

        for step, (img, label) in enumerate(train_loader):

            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Training Block: {}, Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.encoder_config.num_epochs + opt.encoder_config.start_epoch,
                        step,
                        total_step,
                        cur_train_module,
                        time.time() - starttime,
                    )
                )

            starttime = time.time()

            model_input = img.to(opt.device)
            label = label.to(opt.device)

            loss, _, _, accuracy = model(model_input, label, n=cur_train_module)
            loss = torch.mean(loss, 0)  # Take mean over outputs of different GPUs.
            accuracy = torch.mean(accuracy, 0)

            if cur_train_module != model_splits and model_splits > 1:
                loss = loss[cur_train_module].unsqueeze(0)

            model.zero_grad()
            overall_loss = torch.sum(loss)

            overall_loss.backward()
            optimizer.step()

            for idx, cur_losses in enumerate(loss):
                print_loss = cur_losses.item()
                print_acc = accuracy[idx].item()
                loss_epoch[idx] += print_loss
                loss_updates[idx] += 1

                if step % print_idx == 0:
                    print("\t \t Loss: \t \t {:.4f}".format(print_loss))
                    if opt.loss == 1:
                        print("\t \t Accuracy: \t \t {:.4f}".format(print_acc))

            for idx, cur_losses in enumerate(loss):
                wandb.log({f"loss_{idx}": cur_losses}, step=global_step)

            global_step += 1

        if opt.validate:
            validation_loss = validate(opt, model, test_loader)  # Test_loader corresponds to validation set here.
            logs.append_val_loss(validation_loss)

            for i, val_loss in enumerate(validation_loss):
                wandb.log({f"val_loss_{i}": val_loss}, step=global_step)

        logs.append_train_loss([x / loss_updates[idx] for idx, x in enumerate(loss_epoch)])
        logs.create_log(model, epoch=epoch, optimizer=optimizer)


if __name__ == "__main__":

    opt = get_options()
    assert opt.experiment == "vision"

    dataset = opt.encoder_config.dataset.dataset
    arg_parser.create_log_path(opt)

    wandb.init(project=f"SIM_VISION_ENCODER_{dataset}")
    for key, value in vars(opt).items():
        wandb.config[key] = value

    # After initializing the wandb run, get the run id
    run_id = wandb.run.id
    # Save the run id to a file in the logs directory
    with open(os.path.join(opt.log_path, 'wandb_run_id.txt'), 'w') as f:
        f.write(run_id)


    opt.model_type = ModelType.ONLY_ENCODER

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model, optimizer = load_vision_model.load_model_and_optimizer(opt, classifier_config=None)

    logs = logger.Logger(opt)

    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt.encoder_config.dataset,
        purpose_is_unsupervised_learning=True,
    )

    if opt.loss == 1:
        train_loader = supervised_loader

    try:
        # Train the model
        train(opt, model)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)

    wandb.finish()
