from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from pytorch_lightning.trainer import trainer
from torch import Tensor
from arg_parser import arg_parser
from config_code.config_classes import ModelType, OptionsConfig, ClassifierConfig
from decoder.my_data_module import MyDataModule
from linear_classifiers.downstream_classification import ClassifierModel
from options import get_options
from utils.decorators import init_decorator, wandb_resume_decorator, timer_decorator
from utils.utils import get_classif_log_path, get_wandb_audio_classific_key

try:
    import tikzplotlib
except ImportError:
    print("Tikzplotlib not installed. Please install it to save plots as tikz.")


def variances_vs_accuracy_per_input_signal(classifier: ClassifierModel, batch: Tuple[Tensor, Tensor]) -> Tensor:
    """Returns tensor of shape (batch, 2),
    where the first column is the variance and the second column
    is 1 or 0 if the prediction is correct or not."""

    # 1) per input signal, compute multiple predictions
    predictions = classifier.get_predictions_of_all_frames(batch)  # (batch_size, num_frames, num_classes)
    assert len(predictions.shape) == 3

    # 2) compute the variance of the predictions
    variance = predictions.var(dim=1)  # (batch_size, num_classes)
    average_variance = variance.mean(dim=1)  # (batch_size)

    # 3) get mode of the predictions (majority vote)
    predicted_mode, _ = classifier.get_predicted_mode(predictions)  # (batch_size)
    _, labels = batch  # (batch_size)
    accuracy = (predicted_mode == labels)  # (batch_size)
    # 4) compute the accuracy of the mode with the labels
    stack = torch.stack((average_variance, accuracy), dim=1)  # (batch_size, 2)
    return stack


def histogram_of_accuracies(opt: OptionsConfig, classifier_config: ClassifierConfig, title: str,
                            var_vs_accuracy: Tensor):
    variances = var_vs_accuracy[:, 0].numpy()
    accuracies = var_vs_accuracy[:, 1].numpy()  # 0 or 1s

    # Bin the variances
    num_bins = 4  # Adjust the number of bins as needed
    bins = np.linspace(variances.min(), variances.max(), num_bins)
    variance_bins = np.digitize(variances, bins)  # values between 1 ... num_bins

    # Count accuracies for each bin
    accuracy_counts = np.zeros(num_bins)
    for i in range(1, num_bins + 1):
        # Indices where variances fall into the current bin
        indices = np.where(variance_bins == i)[0]
        # Count of 1s in accuracies for the current bin
        accuracy_counts[i - 1] = accuracies[indices].sum()

    # normalize
    accuracy_per_bin = accuracy_counts / len(accuracies)

    bin_width = np.min(np.diff(bins)) * 0.8  # Adjust the 0.8 as needed to change the bar width

    # Create a matplotlib figure and axes
    fig, ax = plt.subplots()
    ax.bar(bins, accuracy_per_bin, width=bin_width)
    ax.set_xlabel("Variance Bins")
    ax.set_ylabel("Accuracy Counts")
    ax.set_title(f"{title} Accuracy Counts per Variance Bin")

    # Log the matplotlib figure to wandb
    if options.use_wandb:
        wandb_section = get_wandb_audio_classific_key(opt, classifier_config)
        wandb.log({f"{wandb_section}_softmax/{title}_plot": wandb.Image(fig)})

    plt.close(fig)  # Close the figure to prevent it from displaying in the notebook or script output


def log_accuracy_vs_variance(opt: OptionsConfig, classifier_config: ClassifierConfig, var_vs_accuracy: Tensor):
    variances = var_vs_accuracy[:, 0].numpy()
    accuracies = var_vs_accuracy[:, 1].numpy()

    # Prepare data for the table
    data = [[variance, accuracy] for variance, accuracy in zip(variances, accuracies)]

    # Create a wandb.Table
    table = wandb.Table(data=data, columns=["Variance", "Accuracy"])

    if options.use_wandb:
        wandb_section = get_wandb_audio_classific_key(opt, classifier_config)
        wandb.log({
            f"{wandb_section}_softmax/variance_vs_accuracy": wandb.plot.scatter(
                table, "Variance", "Accuracy", title="Accuracy vs Variance")})


def distribution_variances_per_wrong_or_correct_prediction(opt: OptionsConfig, classifier_config: ClassifierConfig,
                                                           var_vs_accuracy: Tensor):
    variances = var_vs_accuracy[:, 0].numpy()  # scalar values
    accuracies = var_vs_accuracy[:, 1].numpy()  # 0 or 1s

    fig, ax = plt.subplots()
    colors = ['blue', 'red']  # Colors for correct and wrong predictions
    labels = ['Correct Predictions', 'Wrong Predictions']

    for i, use_correct_predictions in enumerate([1., 0.]):
        indices = np.where(accuracies == use_correct_predictions)[0]
        variances_filtered = variances[indices]
        ax.hist(variances_filtered, bins=20, color=colors[i], alpha=0.5, label=labels[i])

    ax.set_xlabel("Variance")
    ax.set_ylabel("Count")
    ax.set_title("Variance Distribution for Correct vs Wrong Predictions")
    ax.legend()

    # Log the matplotlib figure to wandb
    if options.use_wandb:
        wandb_section = get_wandb_audio_classific_key(opt, classifier_config)
        wandb.log({f"{wandb_section}_softmax/variance_distribution_combined": wandb.Image(fig)})

    plt.show()
    plt.close(fig)


# calculate the accuracy using dataset from different signal-to-noise ratios
def accuracy_at_diff_snr(opt: OptionsConfig, classifier: ClassifierModel,
                         data_module=None) -> Tensor:
    """Returns tensor of shape (snr levels, accuracy)"""
    snr_levels = [-10, -5, 0, 5, 10, 15, 20]
    accuracies: Tensor = torch.tensor([])  # eventually will be shape (snr levels, 1)

    for snr in snr_levels:
        print(f"info: calculating accuracy at snr level: {snr}")
        test_loader = data_module.get_noisy_test_data(opt.device, snr)

        correct = 0
        total = 0

        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)

            # get predictions from classifier
            outputs = classifier.get_predictions_of_pooled_c_softmax(
                batch=(inputs, labels))  # (batch_size, num_classes)
            predicted = torch.argmax(outputs, dim=1)  # (batch_size)

            # calculate the number of correct predictions
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # calculate accuracy for the current SNR level
        accuracy = correct / total
        accuracies = torch.cat((accuracies, torch.tensor([accuracy])), dim=0)

    # assert accuracies is a tensor of shape (snr levels, 1)
    assert len(accuracies) == len(snr_levels)

    total_results = torch.stack((torch.tensor(snr_levels), accuracies),
                                dim=1)  # (snr levels, 2) where total_results[:, 0] is the snr level and total_results[:, 1] is the accuracy
    print(f"info: total results: {total_results}")
    return total_results


@init_decorator  # sets seed and clears cache etc
@wandb_resume_decorator
@timer_decorator
def main(opt: OptionsConfig, classifier_config: ClassifierConfig):
    # loads the log path where classifier is stored
    arg_parser.create_log_path(opt, add_path_var=get_classif_log_path(opt, classifier_config))

    classifier = ClassifierModel(opt, classifier_config)
    classifier.eval()
    data_module = MyDataModule(opt.post_hoc_dataset)

    # load the classifier model
    classifier = classifier.load_classifier(opt)  # update Lightning module as well!!!

    # function to calculate the accuracy at different snr levels
    accuracy_at_diff_snr(opt, classifier, data_module)

    # load the data
    # batch = data_module.get_all_data(opt.device, subset_percentage=0.4)
    # var_vs_accuracy: Tensor = variances_vs_accuracy_per_input_signal(classifier, batch)
    # var_vs_accuracy = var_vs_accuracy.cpu().detach()

    # log_accuracy_vs_variance(opt, opt.classifier_config, var_vs_accuracy)
    # histogram_of_accuracies(opt, opt.classifier_config, "Accuracy", var_vs_accuracy)

    # distribution_variances_per_wrong_or_correct_prediction(opt, opt.classifier_config, var_vs_accuracy)


if __name__ == "__main__":
    options: OptionsConfig = get_options()
    c_config: ClassifierConfig = options.classifier_config
    # options.model_type = ModelType.ONLY_DOWNSTREAM_TASK
    main(options, c_config)
