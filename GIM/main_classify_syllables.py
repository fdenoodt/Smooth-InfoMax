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
    ## TODO: THE DATASET USED FOR CLASSIFICATION IS NOT YET RESHUFFLED, SO STILL OLD TRAIN/TEST
    ####################################

    return OPTIONS, DEVICE, ENCODER, train_loader, test_loader


def train(opt, encoder, classifier, logs, train_loader, test_loader, learning_rate, criterion):
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

        training_losses_epoch = []
        for step, (gt_audio_batch, _, syllable_idx, _) in enumerate(train_loader):
            epoch_printer(step, epoch)

            # (batch_size, 1, 10240)
            gt_audio_batch = gt_audio_batch.to(device)

            cs = encoder(gt_audio_batch)
            c = cs[-1].to(device)

            # (batch_size, l, c)
            c = c.permute(0, 2, 1)  # (b, c, l)

            pooled_c = nn.functional.adaptive_avg_pool1d(c, 1) # (b, c, 1)
            pooled_c = pooled_c.permute(0, 2, 1).reshape(-1, 32) # (b, 1, c) -> (b, c)


            # forward pass
            outputs = classifier(pooled_c)

            # transform syllable_idx to one-hot encoding
            targets = torch.nn.functional.one_hot(syllable_idx, num_classes=10).to(device)

            loss = criterion(outputs, targets) #* (1 / opt["batch_size_multiGPU"])

            # zero the gradients
            optimizer.zero_grad()

            # backward pass and optimization step
            loss.backward()
            optimizer.step()

            training_losses_epoch.append(loss.item())
            # </> end for step

        training_losses.append(np.mean(training_losses_epoch))
        # validation_losses.append(validation_loss(encoder, decoder, test_loader, criterion))

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

    classifier = train(options, encoder, classifier, logs, train_loader, test_loader, lr, criterion)

    # generate_predictions(opt, f"{experiment_name}_experiment", encoder,
    #                      criterion.name, lr, 1, decoder, model_nb=opt['num_epochs'] - 1)

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
    n_classes = 10
    n_features = 32

    classifier = torch.nn.Sequential(torch.nn.Linear(n_features, n_classes))

    criterion = CrossEntropyLoss()
    run_configuration(options, "linear_model", 0.001, criterion, classifier, 20)


    # random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
