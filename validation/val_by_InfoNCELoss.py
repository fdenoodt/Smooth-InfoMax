import time
import torch


def val_by_InfoNCELoss(opt, model, test_loader):
    total_step = len(test_loader)

    model_splits = opt["model_splits"]

    loss_epoch = [0 for i in range(model_splits)]
    starttime = time.time()

    for step, (audio, _, _, _) in enumerate(test_loader):

        model_input = audio.to(opt["device"])

        loss = model(model_input)
        loss = torch.mean(loss, 0)

        loss_epoch += loss.data.cpu().numpy()

    for i in range(model_splits):
        print(
            f"Validation Loss Model {i}: Time (s): {time.time() - starttime:.1f} --- {loss_epoch[i] / total_step:.4f}"
        )

    validation_loss = [x/total_step for x in loss_epoch]
    return validation_loss
