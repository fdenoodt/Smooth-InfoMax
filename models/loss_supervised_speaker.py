import torch.nn as nn
import torch

from config_code.config_classes import OptionsConfig
from data import get_dataloader
from models import loss
from utils import utils


class Speaker_Loss(loss.Loss):
    def __init__(self, opt: OptionsConfig, hidden_dim, calc_accuracy, bias: bool):
        super(Speaker_Loss, self).__init__()

        self.opt = opt
        self.hidden_dim = hidden_dim
        self.calc_accuracy = calc_accuracy
        self.bias = bias

        self.linear_classifier = nn.Sequential(nn.Linear(self.hidden_dim, 251, bias=bias)).to(
            opt.device
        )

        self.label_num = 1
        self.speaker_loss = nn.CrossEntropyLoss()

        # create mapping speaker_id to label
        if torch.cuda.is_available():
            factor = torch.cuda.device_count()
        else:
            factor = 1

        # model is initialized before the dataset is loaded,
        # so we initialize the speaker_id_dict with a separate version of the dataset
        opt.speakers_classifier_config.dataset.batch_size_multiGPU = opt.speakers_classifier_config.dataset.batch_size * factor
        _, train_dataset, _, test_dataset = get_dataloader.get_dataloader(opt.encoder_config.dataset)
        self.speaker_id_dict = {}
        for idx, key in enumerate(train_dataset.speaker_dict):
            self.speaker_id_dict[key] = idx

        # When dataset is a subset of LibriSpeech, speaker_id_dict may not contain all speakers
        # Need to add the missing speakers to the speaker_id_dict
        for idx, key in enumerate(test_dataset.speaker_dict):
            if key not in self.speaker_id_dict:
                self.speaker_id_dict[key] = len(self.speaker_id_dict)

    def get_loss(self, x, z, c, filename, start_idx):
        total_loss, accuracies = self.calc_supervised_speaker_loss(c, filename)
        return total_loss, accuracies

    def calc_supervised_speaker_loss(self, c, filename):
        """
        Calculates the loss for fully supervised training using the provided speaker labels.
        :param c: output of the layer to be trained
        :param filename: filenames of the current files in the batch
        :param start_idx: idx within the audio-files for the current files in the batch
        :return: loss and accuracy
        """

        cur_device = utils.get_device(self.opt, c)

        targets = torch.zeros(len(filename)).long()
        for idx, _ in enumerate(filename):
            speaker_id = filename[idx].split("-")[0]
            targets[idx] = self.speaker_id_dict[speaker_id]
        targets = targets.to(cur_device).squeeze()

        # forward pass
        c = c.permute(0, 2, 1)

        pooled_c = nn.functional.adaptive_avg_pool1d(c, self.label_num)
        pooled_c = pooled_c.permute(0, 2, 1).reshape(-1, self.hidden_dim)

        speaker_out = self.linear_classifier(pooled_c)

        loss = self.speaker_loss(speaker_out, targets)

        accuracy = torch.zeros(1)
        # calculate accuracy
        if self.calc_accuracy:
            _, predicted = torch.max(speaker_out.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            accuracy[0] = correct / total

        return loss, accuracy
