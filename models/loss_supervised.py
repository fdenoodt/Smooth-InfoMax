from typing import Tuple

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

        self.hidden_dim = hidden_dim

        self.calc_accuracy = calc_accuracy
        self.bias = bias

        self.linear_classifier = nn.Linear(self.hidden_dim, num_syllables, bias=True).to(opt.device)

        self.label_num = 1
        self.syllables_loss = nn.CrossEntropyLoss()

    def get_loss(self, x, z, c, targets) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        if self.opt.classifier_config.use_single_frame:  # predict using a single frame
            total_loss, accuracy, mode_accuracy = self.calc_supervised_syllables_loss_subsample(c, targets)
        else:
            total_loss, accuracy, mode_accuracy = self.calc_supervised_syllables_loss(
                c, targets,
            )
        return total_loss, accuracy, mode_accuracy  # mode_accuracies is the same as accuracies for predict_from_single_timeframe=False

    def _compute_accuracy(self, predicted, targets):
        """
        Inp: predicted: tensor of shape (batch_size*num_frames), targets: tensor of shape (batch_size*num_frames),
        or  tensor of shape (batch_size), targets: tensor of shape (batch_size)
        """
        # calculate accuracy
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        return correct / total

    def apply_moving_average_pooling(self, c, window_size=3):
        """
        Applies moving average pooling over all frames with a specified window size,
        independently for each hidden channel.

        Parameters:
        - c: Input tensor with shape (batch_size, hidden_dim, num_frames).
        - window_size: The size of the moving window.

        Returns:
        - pooled_c: Tensor after applying moving average pooling.
        """
        c = c.permute(0, 2, 1)
        batch_size, hidden_dim, num_frames = c.shape

        # Reshape c to treat each hidden channel independently
        c = c.reshape(-1, 1, num_frames)  # shape: (batch_size*hidden_dim, 1, num_frames)

        # Define a 1D convolutional layer for moving average
        moving_avg_filter = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=window_size,
                                      stride=1, padding=0, bias=False).to(c.device)

        # Set uniform weights for the moving average and disable gradient computation
        moving_avg_filter.weight.data.fill_(1.0 / window_size)
        moving_avg_filter.weight.requires_grad = False

        # Apply the moving average filter
        pooled_c = moving_avg_filter(c)

        # Reshape back to original dimensions with moving average applied
        pooled_c = pooled_c.reshape(batch_size, hidden_dim,
                                    -1)  # shape: (batch_size, hidden_dim, num_frames or adjusted)

        pooled_c = pooled_c.permute(0, 2, 1)  # shape: (batch_size, num_frames, hidden_dim)
        return pooled_c

    def calc_supervised_syllables_loss(self, c, targets) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # forward pass
        c = c.permute(0, 2, 1)  # shape: (batch_size, hidden_dim, num_frames) = (128, 256, 16)
        b_size, hidden_dim, num_frames = c.shape
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

        return loss, accuracy, accuracy

    def calc_supervised_syllables_loss_subsample(self, c, targets) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        b_size = c.shape[0]

        syllables_out, num_frames = self.get_predictions_of_all_frames(c)  # (batch_size*num_frames, num_syllables)

        # duplicate targets for each timestep
        targets = targets.repeat(num_frames, 1).T.reshape(-1)
        assert syllables_out.shape[0] == targets.shape[0]

        loss = self.syllables_loss(syllables_out, targets)

        accuracy_single_frame = torch.zeros(1)
        accuracy_mode = torch.zeros(1)

        # calculate accuracy
        if self.calc_accuracy:
            predicted_mode, predicted = self.get_predicted_mode(syllables_out, b_size, num_frames)
            accuracy_single_frame[0] = self._compute_accuracy(predicted.flatten(), targets)

            targets = targets.reshape(b_size, num_frames)
            # all targets are the same, so we can take the first one (or any)
            targets = targets[:, 0]

            accuracy_mode[0] = self._compute_accuracy(predicted_mode, targets)

        return loss, accuracy_single_frame, accuracy_mode

    def get_predicted_mode(self, syllables_out, b_size, num_frames) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inp: syllables_out: tensor of shape (batch_size*num_frames, num_classes)
        Out: predicted_mode: tensor of shape (batch_size), predicted: tensor of shape (batch_size, num_frames)
        """

        # Accuracy of classifier when predicting a single frame
        _, predicted = torch.max(syllables_out.data, 1)  # shape: (batch_size*num_frames, 1)

        # reduce to (batch_size, 1) by taking the most frequent prediction
        predicted = predicted.reshape(b_size, num_frames)
        predicted_mode = torch.mode(predicted, dim=1).values  # shape: (batch_size)

        return predicted_mode, predicted

    def get_predictions_of_all_frames(self, c):
        """
        Get predictions of all frames.
        Returns tensor of shape (batch_size*num_frames, num_syllables). So need to reshape to (batch_size, num_frames, num_syllables) later.
        """
        # moving average (with window size 3) over all frames
        pooled_c = self.apply_moving_average_pooling(c, window_size=3)  # shape: (batch_size, hidden_dim, num_frames)
        c = pooled_c

        # forward pass
        b_size, num_frames, hidden_dim = c.shape
        c = c.reshape(b_size * num_frames, hidden_dim)  # predict on every timestep
        assert c.shape[1] == self.hidden_dim  # verify if 512 or 256, depending on bias
        syllables_out = self.linear_classifier(c)  # shape: (batch_size, 9)

        # softmax
        syllables_out = torch.nn.functional.softmax(syllables_out, dim=1)

        return syllables_out, num_frames
