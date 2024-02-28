import torch
import torch.nn.functional as F

from data import get_dataloader
from models import load_audio_model
from models.loss_supervised_syllables import Syllables_Loss
from options import get_options
from config_code.config_classes import OptionsConfig, ModelType


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def test(context_model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []

    for data, _, target, _ in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    return final_acc, adv_examples


if __name__ == "__main__":
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    torch.manual_seed(42)

    opt: OptionsConfig = get_options()
    opt.model_type = ModelType.ONLY_DOWNSTREAM_TASK
    classifier_config = opt.syllables_classifier_config

    context_model, _ = load_audio_model.load_model_and_optimizer(
        opt,
        classifier_config,
        reload_model=True,
        calc_accuracy=True,
        num_GPU=1,
    )

    n_features = context_model.module.output_dim
    loss = Syllables_Loss(opt, n_features, calc_accuracy=True)



    device = opt.device
    context_model = context_model.to(device)
    context_model.eval()

    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt.syllables_classifier_config.dataset)

    accuracies = []
    examples = []

    for eps in epsilons:
        acc, ex = test(context_model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
