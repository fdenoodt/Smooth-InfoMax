import pandas as pd
import torch
import wandb
from torch import Tensor
from arg_parser import arg_parser
from config_code.config_classes import ModelType, OptionsConfig, ClassifierConfig
from decoder.my_data_module import MyDataModule
from linear_classifiers.downstream_classification import ClassifierModel
from models.load_audio_model import load_classifier
from options import get_options
from utils.decorators import init_decorator, wandb_resume_decorator, timer_decorator
from utils.utils import get_classif_log_path


def variances_vs_accuracy_per_input_signal(classifier: ClassifierModel, batch: Tensor) -> Tensor:
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
    classifier.classifier = load_classifier(opt, classifier.classifier)  # update Lightning module as well!!!

    # load the data
    batch = data_module.get_batch(opt.device)

    var_vs_accuracy: Tensor = variances_vs_accuracy_per_input_signal(classifier, batch)
    var_vs_accuracy = var_vs_accuracy.cpu().detach()
    variances = var_vs_accuracy[:, 0].numpy()
    accuracies = var_vs_accuracy[:, 1].numpy()

    # 

    # Prepare data for the table
    data = [[variance, accuracy] for variance, accuracy in zip(variances, accuracies)]

    # Create a wandb.Table
    table = wandb.Table(data=data, columns=["Variance", "Accuracy"])

    # Log the table with a custom line plot
    wandb.log({
        "variance_vs_accuracy": wandb.plot.scatter(table, "Variance", "Accuracy", title="Accuracy vs Variance")
    })

    # graph the variance vs accuracy to wandb.
    # x axis is variance, y axis is accuracy
    # TODO: implement this
    # data_to_log = [{"variance": float(var_vs_accuracy[i, 0]), "accuracy": float(var_vs_accuracy[i, 1])} for i in
    #                range(var_vs_accuracy.shape[0])]
    #
    # # Log the data
    # for data_point in data_to_log:
    #     wandb.log(data_point)



if __name__ == "__main__":
    options: OptionsConfig = get_options()
    c_config: ClassifierConfig = options.classifier_config
    options.model_type = ModelType.ONLY_DOWNSTREAM_TASK
    main(options, c_config)
