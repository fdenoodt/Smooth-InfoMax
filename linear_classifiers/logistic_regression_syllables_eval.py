import torch
from models.full_model import FullModel
from options import get_options
from data import get_dataloader
from config_code.config_classes import OptionsConfig, ModelType
from arg_parser import arg_parser
from models import load_audio_model
from models.loss_supervised_syllables import Syllables_Loss
import torch.nn as nn


def main():
    opt: OptionsConfig = get_options()
    opt.model_type = ModelType.ONLY_DOWNSTREAM_TASK

    arg_parser.create_log_path(opt, add_path_var="linear_model_syllables")

    context_model, _ = load_audio_model.load_model_and_optimizer(
        opt,
        opt.syllables_classifier_config,
        reload_model=True,
        calc_accuracy=True,
        num_GPU=1,
    )

    # the classifier is a part of the loss function
    n_features = context_model.module.output_dim
    n_labels = 9
    syllables_loss = Syllables_Loss(opt, hidden_dim=n_features, calc_accuracy=True)

    # Load the trained model
    model_path = opt.log_path + '/model_0.ckpt'
    syllables_loss.load_state_dict(torch.load(model_path))

    # Load a few data points
    train_loader, _, _, _ = get_dataloader.get_dataloader(opt.syllables_classifier_config.dataset, shuffle=True)

    # Use the model to make predictions
    context_model.eval()
    syllables_loss.linear_classifier.eval()

    with torch.no_grad():
        for i, (audio, _, label, _) in enumerate(train_loader):
            audio = audio.to(opt.device)

            # get latent representations for current audio
            model_input = audio.to(opt.device)
            full_model: FullModel = context_model.module
            z = full_model.forward_through_all_modules(model_input) # shape: (128, 16, 256)
            z = z.permute(0, 2, 1) # shape: (128, 256, 16)

            pooled_z = nn.functional.adaptive_avg_pool1d(z, 1) # shape: (128, 256, 1)
            pooled_z = pooled_z.permute(0, 2, 1).reshape(-1, n_features)  # shape: (128, 256)
            # = avg over all frames

            # forward pass -> shape: (128, 9)
            outputs = syllables_loss.linear_classifier(pooled_z)

            # Get the predicted class for each frame in each sample
            _, predicted = torch.max(outputs.data, 1)

            print(f'Predicted frames for a data point: {predicted[0].item()}, True: {label[0].item()}')

            # Stop after a few data points
            if i >= 5:
                break


if __name__ == "__main__":
    main()
