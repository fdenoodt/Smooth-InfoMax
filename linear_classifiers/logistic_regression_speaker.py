import torch
import time
import numpy as np

## own modules
from config_code.config_classes import OptionsConfig, ModelType, Dataset
from models.full_model import FullModel
from models.loss_supervised_speaker import Speaker_Loss
from options import get_options
from data import get_dataloader
from utils import logger
from arg_parser import arg_parser
from models import load_audio_model, loss_supervised_speaker
from utils.utils import set_seed


def train(opt: OptionsConfig, context_model, loss: Speaker_Loss, logs: logger.Logger, train_loader, optimizer):
    total_step = len(train_loader)
    print_idx = 100

    num_epochs = opt.speakers_classifier_config.num_epochs

    for epoch in range(num_epochs):
        loss_epoch = 0
        acc_epoch = 0
        for i, (audio, filename, _, audio_idx) in enumerate(train_loader):

            starttime = time.time()

            loss.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(opt.device)

            with torch.no_grad():
                full_model: FullModel = context_model.module
                z = full_model.forward_through_all_modules(model_input)
            z = z.detach()

            # forward pass
            total_loss, accuracies = loss.get_loss(model_input, z, z, filename, audio_idx)

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            sample_loss = total_loss.item()
            accuracy = accuracies.item()

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


def test(opt, context_model, loss, data_loader):
    loss.eval()
    accuracy = 0
    loss_epoch = 0

    with torch.no_grad():
        for i, (audio, filename, _, audio_idx) in enumerate(data_loader):

            loss.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(opt.device)

            with torch.no_grad():
                full_model: FullModel = context_model.module
                z = full_model.forward_through_all_modules(model_input)

            z = z.detach()

            # forward pass
            total_loss, step_accuracy = loss.get_loss(model_input, z, z, filename, audio_idx)

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
    return loss_epoch, accuracy


def main(model_type: ModelType = ModelType.ONLY_DOWNSTREAM_TASK):
    opt: OptionsConfig = get_options()
    opt.model_type = model_type

    classifier_config = opt.speakers_classifier_config

    assert opt.speakers_classifier_config is not None, "Classifier config is not set"
    assert opt.model_type in [ModelType.FULLY_SUPERVISED,
                              ModelType.ONLY_DOWNSTREAM_TASK], "Model type not supported"
    assert opt.speakers_classifier_config.dataset.dataset in [Dataset.LIBRISPEECH,
                                                              Dataset.LIBRISPEECH_SUBSET], "Dataset not supported"

    arg_parser.create_log_path(opt, add_path_var="linear_model_speaker")

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

    n_features = context_model.module.output_dim

    loss: Speaker_Loss = loss_supervised_speaker.Speaker_Loss(
        opt, n_features, calc_accuracy=True
    )

    learning_rate = opt.speakers_classifier_config.learning_rate
    optimizer = torch.optim.Adam(loss.parameters(), lr=learning_rate)

    # load dataset
    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt.speakers_classifier_config.dataset)

    logs = logger.Logger(opt)
    accuracy = 0

    try:
        # Train the model
        train(opt, context_model, loss, logs, train_loader, optimizer)

        # Test the model
        result_loss, accuracy = test(opt, context_model, loss, test_loader)

    except KeyboardInterrupt:
        print("Training interrupted, saving log files")

    logs.create_log(loss, accuracy=accuracy, final_test=True, final_loss=result_loss)


if __name__ == "__main__":
    main()
