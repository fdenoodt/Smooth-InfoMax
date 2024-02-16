import torch.nn as nn
import torch
import numpy as np

from config_code.config_classes import OptionsConfig
from models import loss
from utils import utils


class InfoNCE_Loss(loss.Loss):
    def __init__(self, opt: OptionsConfig, hidden_dim, enc_hidden, calc_accuracy, prediction_step):
        super(InfoNCE_Loss, self).__init__()

        self.opt = opt
        self.hidden_dim = hidden_dim
        self.enc_hidden = enc_hidden
        self.neg_samples = self.opt.encoder_config.negative_samples
        self.calc_accuracy = calc_accuracy
        self.prediction_step = prediction_step

        self.predictor = nn.Linear(
            self.hidden_dim, self.enc_hidden * self.prediction_step, bias=False
        )

        if self.opt.encoder_config.subsample:
            self.subsample_win = 128

        self.loss = nn.LogSoftmax(dim=1)

    def get_loss(self, z, c):

        full_z = z

        """
        Subsample: 
        positive samples are restricted to this subwindow to reduce the number of calculations for the loss, 
        negative samples can still come from any point of the input sequence (full_z)
        """
        if c.size(1) > self.subsample_win:
            seq_begin = np.random.randint(
                0, c.size(1) - self.subsample_win)
            c = c[:, seq_begin: seq_begin + self.subsample_win, :]
            z = z[:, seq_begin: seq_begin + self.subsample_win, :]

        Wc = self.predictor(c)
        total_loss, accuracies = self.calc_InfoNCE_loss(Wc, z, full_z)
        return total_loss, accuracies

    def broadcast_batch_length(self, input_tensor):
        """
        broadcasts the given tensor in a consistent way, such that it can be applied to different inputs and
        keep their indexing compatible
        :param input_tensor: tensor to be broadcasted, generally of shape B x L x C
        :return: reshaped tensor of shape (B*L) x C
        """
        assert input_tensor.size(0) == self.opt.encoder_config.dataset.batch_size
        assert len(input_tensor.size()) == 3

        return input_tensor.reshape(-1, input_tensor.size(2))

    def get_pos_sample_f(self, Wc_k, z_k):
        """
        calculate the output of the log-bilinear model for the positive samples, i.e. where z_k is the actual
        encoded future that had to be predicted
        :param Wc_k: prediction of the network for the encoded future at time-step t+k (dimensions: (B*L) x C)
        :param z_k: encoded future at time-step t+k (dimensions: (B*L) x C)
        :return: f_k, output of the log-bilinear model (without exp, as this is part of the log-softmax function)
        """
        Wc_k = Wc_k.unsqueeze(1)
        z_k = z_k.unsqueeze(2)
        f_k = torch.squeeze(torch.matmul(Wc_k, z_k), 1)
        return f_k

    def get_neg_z(self, z, cur_device):
        """
        scramble z to retrieve negative samples, i.e. z values that should not be predicted by the model
        :param z: unshuffled z as output by the model
        :return: z_neg - shuffled z to be used for negative sampling
                shuffling params rand_neg_idx, rand_offset for testing this function
        """

        """ randomly selecting from all z values; 
            can cause positive samples to be selected as negative samples as well 
            (but probability is <0.1% in our experiments)
            done once for all time-steps, much faster
        """
        z = self.broadcast_batch_length(z)
        z_neg = torch.stack(
            [
                torch.index_select(
                    z, 0, torch.randperm(z.size(0), device=cur_device)
                )
                for i in range(self.neg_samples)
            ],
            2,
        )
        rand_neg_idx = None
        rand_offset = None

        return z_neg, rand_neg_idx, rand_offset

    def get_neg_samples_f(self, Wc_k, z_neg=None, k=None):
        """
        calculate the output of the log-bilinear model for the negative samples. For this, we get z_k_neg from z_k
        by randomly shuffling the indices.
        :param Wc_k: prediction of the network for the encoded future at time-step t+k (dimensions: (B*L) x C)
        :param z_k: encoded future at time-step t+k (dimensions: (B*L) x C)
        :return: f_k, output of the log-bilinear model (without exp, as this is part of the log-softmax function)
        """
        Wc_k = Wc_k.unsqueeze(1)

        """
            by shortening z_neg from the front, we get different negative samples
            for every prediction-step without having to re-sample;
            this might cause some correlation between the losses within a batch
            (e.g. negative samples for projecting from z_t to z_(t+k+1) 
            and from z_(t+1) to z_(t+k) are the same)                
        """
        z_k_neg = z_neg[z_neg.size(0) - Wc_k.size(0):, :, :]

        f_k = torch.squeeze(torch.matmul(Wc_k, z_k_neg), 1)

        return f_k

    def calc_InfoNCE_loss(self, Wc, z, full_z=None):
        """
        calculate the loss based on the model outputs Wc (the prediction) and z (the encoded future)
        :param Wc: output of the predictor, where W are the weights for the different timesteps and
        c the latent representation (either from the autoregressor, if use_autoregressor=True,
        or from the encoder otherwise) - dimensions: (B, L, C*self.prediction_step)
        :param z: encoded future - output of the encoder - dimensions: (B, L, C)
        :return: total_loss - average loss over all samples, timesteps and prediction steps in the batch
                    accuracies - average accuracies over all samples, timesteps and predictions steps in the batch
        """
        seq_len = z.size(1)

        cur_device = utils.get_device(self.opt, Wc)

        total_loss = 0
        batch_size = self.opt.encoder_config.dataset.batch_size

        accuracies = torch.zeros(self.prediction_step, 1)
        true_labels = torch.zeros(
            (seq_len * batch_size,), device=cur_device
        ).long()

        z_neg, _, _ = self.get_neg_z(full_z, cur_device)

        for k in range(1, self.prediction_step + 1):
            z_k = z[:, k:, :]
            Wc_k = Wc[:, :-k, (k - 1) * self.enc_hidden: k * self.enc_hidden]

            z_k = self.broadcast_batch_length(z_k)
            Wc_k = self.broadcast_batch_length(Wc_k)

            pos_samples = self.get_pos_sample_f(Wc_k, z_k)
            neg_samples = self.get_neg_samples_f(Wc_k, z_neg, k)

            # concatenate positive and negative samples
            results = torch.cat((pos_samples, neg_samples), 1)
            loss = self.loss(results)[:, 0]

            total_samples = (seq_len - k) * batch_size
            loss = -loss.sum() / total_samples
            total_loss += loss

            # calculate accuracy
            if self.calc_accuracy:
                predicted = torch.argmax(results, 1)
                correct = (
                    (predicted == true_labels[: (seq_len - k) * batch_size])
                    .sum()
                    .item()
                )
                accuracies[k - 1] = correct / total_samples

        total_loss /= self.prediction_step
        accuracies = torch.mean(accuracies)

        return total_loss, accuracies
