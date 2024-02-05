import time
import torch

from configs.config_classes import OptionsConfig


def val_by_InfoNCELoss(opt: OptionsConfig, model, test_loader):
    total_step = len(test_loader)

    nb_modules = len(opt.encoder_config.architecture.modules)

    loss_epoch = [0 for i in range(nb_modules)]
    starttime = time.time()

    for step, (audio, _, _, _) in enumerate(test_loader):
        model_input = audio.to(opt.device)

        loss = model(model_input)
        loss = torch.mean(loss, 0)

        loss_epoch += loss.data.cpu().numpy()

    for i in range(nb_modules):
        print(
            f"Validation Loss Module {i}: Time (s): {time.time() - starttime:.1f} --- {loss_epoch[i] / total_step:.4f}"
        )

    validation_loss = [x / total_step for x in loss_epoch]
    return validation_loss
