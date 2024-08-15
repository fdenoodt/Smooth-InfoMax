import torch
import time
import numpy as np

## own modules
from config_code.config_classes import OptionsConfig, ModelType, Dataset, ClassifierConfig
from linear_classifiers.logistic_regression import get_z
from models.full_model import FullModel
from models.loss_supervised_speaker import Speaker_Loss
from options import get_options
from data import get_dataloader
from utils import logger
from arg_parser import arg_parser
from models import load_audio_model, loss_supervised_speaker
from utils.utils import set_seed, retrieve_existing_wandb_run_id, get_audio_libri_classific_key, get_classif_log_path
import wandb


def train(opt: OptionsConfig, context_model, loss: Speaker_Loss, logs: logger.Logger, train_loader, optimizer, bias):
    total_step = len(train_loader)
    print_idx = 100

    num_epochs = opt.speakers_classifier_config.num_epochs
    global_step = 0

    for epoch in range(num_epochs):
        loss_epoch = 0
        acc_epoch = 0
        for i, (audio, filename, _, audio_idx) in enumerate(train_loader):
            # starttime = time.time()
            #
            # loss.zero_grad()
            #
            # ### get latent representations for current audio
            # model_input = audio.to(opt.device)
            #
            # with torch.no_grad():
            #     full_model: FullModel = context_model.module
            #     z = full_model.forward_through_all_modules(model_input)
            # z = z.detach()
            #
            # # forward pass
            # total_loss, accuracies = loss.get_loss(model_input, z, z, filename, audio_idx)

            audio = audio.to(opt.device)
            # label = label.to(opt.device)

            starttime = time.time()
            loss.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(opt.device)
            z = get_z(opt, context_model, model_input,
                      regression=bias,
                      which_module=opt.syllables_classifier_config.encoder_module,
                      which_layer=opt.syllables_classifier_config.encoder_layer
                      )

            # forward pass
            # total_loss, accuracies = loss.get_loss(model_input, z, z, label)
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

            if opt.use_wandb:
                wandb_section = get_audio_libri_classific_key(
                    "speakers",
                    module_nb=opt.speakers_classifier_config.encoder_module,
                    layer_nb=opt.speakers_classifier_config.encoder_layer,
                    bias=opt.speakers_classifier_config.bias)
                wandb.log({f"{wandb_section}/Train Loss": sample_loss,
                           f"{wandb_section}/Train Accuracy": accuracy})

            global_step += 1

        logs.append_train_loss([loss_epoch / total_step])


def test(opt, context_model, loss, data_loader, bias):
    loss.eval()
    accuracy = 0
    loss_epoch = 0

    with torch.no_grad():
        for i, (audio, filename, _, audio_idx) in enumerate(data_loader):

            loss.zero_grad()

            ### get latent representations for current audio
            model_input = audio.to(opt.device)

            # with torch.no_grad():
            #     full_model: FullModel = context_model.module
            #     z = full_model.forward_through_all_modules(model_input)

            with torch.no_grad():
                z = get_z(opt, context_model, model_input, regression=bias,
                          which_module=opt.syllables_classifier_config.encoder_module,
                          which_layer=opt.syllables_classifier_config.encoder_layer)

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

    if opt.use_wandb:
        wandb_section = get_audio_libri_classific_key(
            "speakers",
            module_nb=opt.speakers_classifier_config.encoder_module,
            layer_nb=opt.speakers_classifier_config.encoder_layer,
            bias=opt.speakers_classifier_config.bias)
        wandb.log({f"{wandb_section}/Test Accuracy": accuracy})

    return loss_epoch, accuracy


def main(model_type: ModelType = ModelType.ONLY_DOWNSTREAM_TASK):
    opt: OptionsConfig = get_options()
    opt.model_type = model_type

    classifier_config: ClassifierConfig = opt.speakers_classifier_config
    bias = classifier_config.bias

    if opt.use_wandb:
        run_id, project_name = retrieve_existing_wandb_run_id(opt)
        wandb.init(id=run_id, resume="allow", project=project_name, entity=opt.wandb_entity)

    # on which module to train the classifier (default: -1, last module)
    classif_module: int = classifier_config.encoder_module
    classif_layer: int = classifier_config.encoder_layer
    classif_path = get_classif_log_path(classifier_config, classif_module, classif_layer, bias)
    arg_parser.create_log_path(
        opt, add_path_var=classif_path)
    # arg_parser.create_log_path(opt, add_path_var="linear_model_speaker")

    assert opt.speakers_classifier_config is not None, "Classifier config is not set"
    assert opt.model_type in [ModelType.FULLY_SUPERVISED,
                              ModelType.ONLY_DOWNSTREAM_TASK], "Model type not supported"
    assert opt.speakers_classifier_config.dataset.dataset in [Dataset.LIBRISPEECH,
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

    regr_hidden_dim = opt.encoder_config.architecture.modules[0].regressor_hidden_dim
    cnn_hidden_dim = opt.encoder_config.architecture.modules[0].cnn_hidden_dim

    if bias:
        n_features = regr_hidden_dim
    else:
        n_features = cnn_hidden_dim

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
        if opt.train:
            train(opt, context_model, loss, logs, train_loader, optimizer, bias)

        # Test the model
        result_loss, accuracy = test(opt, context_model, loss, test_loader, bias)

    except KeyboardInterrupt:
        print("Training interrupted, saving log files")

    logs.create_log(loss, accuracy=accuracy, final_test=True, final_loss=result_loss)

    if opt.use_wandb:
        wandb.finish()

    print("DONE")


if __name__ == "__main__":
    main()
