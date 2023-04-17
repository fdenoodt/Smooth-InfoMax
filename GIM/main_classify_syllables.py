from eval_autoencoder import generate_predictions
from arg_parser import arg_parser
import helper_functions
import decoder_architectures
import importlib
import time
import numpy as np
import random
import torch
from GIM_encoder import GIM_Encoder
from data import get_dataloader
from options_classify_syllables import get_options
import matplotlib.pyplot as plt
from decoder_architectures import *
from helper_functions import *
from utils import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup():
    OPTIONS = get_options()
    DEVICE = OPTIONS["device"]

    CPC_MODEL_PATH = OPTIONS["cpc_model_path"]

    ENCODER = GIM_Encoder(OPTIONS, path=CPC_MODEL_PATH)
    ENCODER.encoder.eval()

    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        OPTIONS, dataset="de_boer_sounds", split_and_pad=True, train_noise=False, shuffle=True)

    ####################################
    # TODO: THE DATASET USED FOR CLASSIFICATION IS NOT YET RESHUFFLED, SO STILL OLD TRAIN/TEST
    ####################################

    return OPTIONS, DEVICE, ENCODER, train_loader, test_loader


def validation_loss(GIM_encoder, model, test_loader, criterion):
    # based on GIM/ChatGPT
    total_step = len(test_loader)

    val_l = 0.0
    starttime = time.time()

    for step, (gt_audio_batch, _, syllable_idx, _) in enumerate(test_loader):

        loss, accuracy = forward_and_loss(
            encoder, classifier, gt_audio_batch, syllable_idx, criterion, detach=True)
        val_l += loss.item() / total_step

    print(f"Validation Loss: Time (s): {time.time() - starttime:.1f} --- L: {val_l:.4f}, A: {accuracy:.2f}")

    # validation_l = np.mean(loss_epoch)
    return val_l


def forward_and_loss(encoder, classifier, gt_audio_batch, syllable_idx, criterion, detach):
    # (batch_size, 1, 10240)
    gt_audio_batch = gt_audio_batch.to(device)

    cs = encoder(gt_audio_batch)
    c = cs[-1].to(device)

    # (batch_size, l, c)
    c = c.permute(0, 2, 1)  # (b, c, l)

    pooled_c = nn.functional.adaptive_avg_pool1d(c, 1)  # (b, c, 1)
    pooled_c = pooled_c.permute(0, 2, 1).reshape(-1, 32)  # (b, 1, c) -> (b, c)

    if detach:
        classifier.eval()
        with torch.no_grad():
            outputs = classifier(pooled_c)
        classifier.train()
    else:
        outputs = classifier(pooled_c)

    # transform syllable_idx to one-hot encoding
    targets = torch.nn.functional.one_hot(syllable_idx, num_classes=N_CLASSES).to(device)
    loss = criterion(outputs, targets)

    accuracy, = utils.accuracy(outputs.data, syllable_idx.to(device))

    return loss, accuracy


def train(opt, encoder, classifier, logs, train_loader, test_loader, learning_rate, criterion):
    total_step = len(train_loader)

    epoch_printer = EpochPrinter(train_loader, learning_rate, criterion)
    log_handler = LogHandler(opt, logs, train_loader,
                             criterion, encoder, learning_rate)

    classifier.to(device)
    classifier.train()

    optimizer = torch.optim.Adam(classifier.parameters(
    ), lr=learning_rate, weight_decay=1e-5)  # 1.5 * 10^-2 = 1.5/100

    training_losses = []
    validation_losses = []

    for epoch in range(opt["start_epoch"], opt["num_epochs"] + opt["start_epoch"]):
        training_l = 0.0
        for step, (gt_audio_batch, _, syllable_idx, _) in enumerate(train_loader):
            epoch_printer(step, epoch)

            loss, accuracy = forward_and_loss(
                encoder, classifier, gt_audio_batch, syllable_idx, criterion, detach=False)

            # zero the gradients
            optimizer.zero_grad()
            
            # backward pass and optimization step
            loss.backward()
            optimizer.step()

            training_l += (loss.item() / total_step)
            # </> end for step



        validation_l = validation_loss(encoder, classifier, test_loader, criterion)

        training_losses.append(training_l)
        validation_losses.append(validation_l)

        log_handler(classifier, epoch, optimizer,
                    training_losses, validation_losses)

    # </> end epoch

    return classifier


def run_configuration(options, experiment_name, lr, criterion, classifier, num_epochs):
    torch.cuda.empty_cache()

    options['experiment'] = experiment_name
    options['save_dir'] = f'{experiment_name}_experiment'
    options['log_path'] = options['root_logs'] + "/CLASSIFIER"
    options['log_path_latent'] = options['log_path'] + "/latent_space"

    options['batch_size'] = 8
    options['num_epochs'] = num_epochs

    arg_parser.create_log_path(options)

    create_log_dir(opt['log_path'])

    logs = logger.Logger(options)

    classifier = train(options, encoder, classifier, logs,
                       train_loader, test_loader, lr, criterion)

    torch.cuda.empty_cache()


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "CrossEntrop Loss"
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, batch_inputs, batch_targets):
        assert batch_inputs.shape == batch_targets.shape
        # fix: RuntimeError: Expected floating point type for target with class probabilities, got Long
        batch_targets = batch_targets.float()

        return self.cross_entropy_loss(batch_inputs, batch_targets)


if __name__ == "__main__":

    options, device, encoder, train_loader, test_loader = setup()

    # create linear classifier
    N_CLASSES = 9
    n_features = 32

    classifier = torch.nn.Sequential(torch.nn.Linear(n_features, N_CLASSES))

    criterion = CrossEntropyLoss()
    run_configuration(options, "linear_model", 0.001,
                      criterion, classifier, 20)

    # random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
