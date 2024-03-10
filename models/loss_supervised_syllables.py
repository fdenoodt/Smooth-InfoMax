import torch.nn as nn
import torch

from config_code.config_classes import OptionsConfig
from data import get_dataloader
from models import loss
from utils import utils


class Syllables_Loss(loss.Loss):
    # also used for vowels
    def __init__(self, opt: OptionsConfig, hidden_dim, calc_accuracy, num_syllables: int, bias: bool):
        super(Syllables_Loss, self).__init__()

        """ 
        WARNING: bias = False is only used for vowel classifier on the ConvLayer. It's not supported beyond that (eg regression layer).
        It is only used for the latent space analysis, not used for performance evaluation.
        """

        self.opt = opt

        # TODO: update in logistic_regression_vowel etc
        #self.hidden_dim = hidden_dim if bias else hidden_dim * 2  # 512 is the output of the ConvLayer, only used for space analysis
        self.hidden_dim = hidden_dim

        self.calc_accuracy = calc_accuracy
        self.bias = bias

        self.linear_classifier = nn.Linear(self.hidden_dim, num_syllables, bias=True).to(opt.device)

        self.label_num = 1
        self.syllables_loss = nn.CrossEntropyLoss()

    def get_loss(self, x, z, c, targets):
        total_loss, accuracies = self.calc_supervised_syllables_loss(
            c, targets,
        )
        return total_loss, accuracies

    def calc_supervised_syllables_loss(self, c, targets):
        # forward pass
        c = c.permute(0, 2, 1)  # shape: (batch_size, hidden_dim, num_frames) = (128, 256, 16)

        # avg over all frames
        pooled_c = nn.functional.adaptive_avg_pool1d(c, self.label_num)  # shape: (batch_size, hidden_dim, 1)

        assert c.shape[1] == self.hidden_dim  # verify if 512 or 256, depending on bias

        pooled_c = pooled_c.permute(0, 2, 1).reshape(-1, self.hidden_dim)  # shape: (batch_size, hidden_dim)

        syllables_out = self.linear_classifier(pooled_c)  # shape: (batch_size, 9)

        assert syllables_out.shape[0] == targets.shape[0]

        loss = self.syllables_loss(syllables_out, targets)

        accuracy = torch.zeros(1)
        # calculate accuracy
        if self.calc_accuracy:
            _, predicted = torch.max(syllables_out.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            accuracy[0] = correct / total

        return loss, accuracy
