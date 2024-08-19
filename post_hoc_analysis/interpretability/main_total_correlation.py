# import torch
# import time
# import numpy as np
#
# ## own modules
# from config_code.config_classes import OptionsConfig, ModelType, Dataset, ClassifierConfig
# from linear_classifiers.logistic_regression import get_z
# from models.full_model import FullModel
# from models.loss_supervised_speaker import Speaker_Loss
# from options import get_options
# from data import get_dataloader
# from utils import logger
# from arg_parser import arg_parser
# from models import load_audio_model, loss_supervised_speaker
# from utils.utils import set_seed, retrieve_existing_wandb_run_id, get_audio_libri_classific_key, get_classif_log_path
# import wandb
#
#
# def train(opt: OptionsConfig, context_model, loss: Speaker_Loss, logs: logger.Logger, train_loader, optimizer, bias):
#     total_step = len(train_loader)
#     print_idx = 100
#
#     num_epochs = opt.speakers_classifier_config.num_epochs
#     global_step = 0
#
#     for epoch in range(num_epochs):
#         loss_epoch = 0
#         acc_epoch = 0
#         for i, (audio, filename, _, audio_idx) in enumerate(train_loader):
#             audio = audio.to(opt.device)
#             starttime = time.time()
#             loss.zero_grad()
#
#             ### get latent representations for current audio
#             model_input = audio.to(opt.device)
#             z = get_z(opt, context_model, model_input,
#                       regression=bias,
#                       which_module=opt.speakers_classifier_config.encoder_module,
#                       which_layer=opt.speakers_classifier_config.encoder_layer
#                       )
#
#             # forward pass
#             # total_loss, accuracies = loss.get_loss(model_input, z, z, label)
#             total_loss, accuracies = loss.get_loss(model_input, z, z, filename, audio_idx)
#
#             # Backward and optimize
#             optimizer.zero_grad()
#             total_loss.backward()  # compute gradients
#
#             # optional: gradient clipping
#             if opt.speakers_classifier_config.gradient_clipping != 0.0:
#                 torch.nn.utils.clip_grad_norm_(loss.parameters(), opt.speakers_classifier_config.gradient_clipping)
#
#             optimizer.step()
#
#             sample_loss = total_loss.item()
#             accuracy = accuracies.item()
#
#             if i % print_idx == 0:
#                 print(
#                     "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Accuracy: {:.4f}, Loss: {:.4f}".format(
#                         epoch + 1,
#                         num_epochs,
#                         i,
#                         total_step,
#                         time.time() - starttime,
#                         accuracy,
#                         sample_loss,
#                     )
#                 )
#                 starttime = time.time()
#
#             loss_epoch += sample_loss
#             acc_epoch += accuracy
#
#             if opt.use_wandb:
#                 wandb_section = get_audio_libri_classific_key(
#                     "speakers",
#                     module_nb=opt.speakers_classifier_config.encoder_module,
#                     layer_nb=opt.speakers_classifier_config.encoder_layer,
#                     bias=opt.speakers_classifier_config.bias,
#                     deterministic_encoder=opt.encoder_config.deterministic)
#                 wandb.log({f"{wandb_section}/Train Loss": sample_loss,
#                            f"{wandb_section}/Train Accuracy": accuracy})
#
#             global_step += 1
#
#         logs.append_train_loss([loss_epoch / total_step])
#
#
# def test(opt, context_model, loss, data_loader, bias):
#     loss.eval()
#     accuracy = 0
#     loss_epoch = 0
#
#     with torch.no_grad():
#         for i, (audio, filename, _, audio_idx) in enumerate(data_loader):
#
#             loss.zero_grad()
#
#             ### get latent representations for current audio
#             model_input = audio.to(opt.device)
#
#             with torch.no_grad():
#                 z = get_z(opt, context_model, model_input, regression=bias,
#                           which_module=opt.speakers_classifier_config.encoder_module,
#                           which_layer=opt.speakers_classifier_config.encoder_layer)
#
#             z = z.detach()
#
#             # forward pass
#             total_loss, step_accuracy = loss.get_loss(model_input, z, z, filename, audio_idx)
#
#             accuracy += step_accuracy.item()
#             loss_epoch += total_loss.item()
#
#             if i % 10 == 0:
#                 print(
#                     "Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}".format(
#                         i, len(data_loader), loss_epoch / (i + 1), accuracy / (i + 1)
#                     )
#                 )
#
#     accuracy = accuracy / len(data_loader)
#     loss_epoch = loss_epoch / len(data_loader)
#     print("Final Testing Accuracy: ", accuracy)
#     print("Final Testing Loss: ", loss_epoch)
#
#     if opt.use_wandb:
#         wandb_section = get_audio_libri_classific_key(
#             "speakers",
#             module_nb=opt.speakers_classifier_config.encoder_module,
#             layer_nb=opt.speakers_classifier_config.encoder_layer,
#             bias=opt.speakers_classifier_config.bias,
#             deterministic_encoder=opt.encoder_config.deterministic)
#         wandb.log({f"{wandb_section}/Test Accuracy": accuracy})
#
#     return loss_epoch, accuracy
#
#
# def main(model_type: ModelType = ModelType.ONLY_DOWNSTREAM_TASK):
#     opt: OptionsConfig = get_options()
#
#     # WARNING: THE ENCODER'S OUTPUT IS BASED ON `speakers_classifier_config`
#     # No classifier is trained here.
#     classifier_config: ClassifierConfig = opt.speakers_classifier_config
#     classif_module: int = classifier_config.encoder_module
#     classif_layer: int = classifier_config.encoder_layer
#
#     if opt.use_wandb:
#         run_id, project_name = retrieve_existing_wandb_run_id(opt)
#         wandb.init(id=run_id, resume="allow", project=project_name, entity=opt.wandb_entity)
#
#     arg_parser.create_log_path(opt, add_path_var="total_correlation")
#
#     assert opt.speakers_classifier_config.dataset.dataset in [
#         Dataset.DE_BOER,
#         Dataset.LIBRISPEECH,
#         Dataset.LIBRISPEECH_SUBSET], "Dataset not supported"
#
#     # random seeds
#     set_seed(opt.seed)
#
#     ## load model
#     context_model, _ = load_audio_model.load_model_and_optimizer(
#         opt,
#         classifier_config,
#         reload_model=True,
#         calc_accuracy=True,
#         num_GPU=1,
#     )
#     context_model.eval()
#
#     # load dataset
#     train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt.speakers_classifier_config.dataset)
#
#     # logs = logger.Logger(opt)
#     total_correlation(opt, context_model, test_loader, classif_module, classif_layer)
#
#     if opt.use_wandb:
#         wandb.finish()
#
#
# if __name__ == "__main__":
#     main()


import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity
import time

if __name__ == "__main__":

    start = time.time()
    # Assuming you have an encoder and a dataset ready
    latent_codes = None
    for x in range(2):  # 15 batches
        h = 3
        z = torch.randn(64, h, 7, 7)
        # should be of shape (batch_size * height * width, num_features)
        z = z.permute(0, 2, 3, 1)  # (64, 7, 7, 10)
        z = z.reshape(-1, h)  # (3136, 10)
        if latent_codes is None:
            latent_codes = z
        else:
            latent_codes = torch.cat((latent_codes, z), dim=0)
        # latent_codes.append(z.cpu().numpy())
    # latent_codes = np.concatenate(latent_codes, axis=0)

    # to numpy
    latent_codes = latent_codes.cpu().numpy()

    # Estimate marginal densities
    marginal_kdes = []
    for i in range(latent_codes.shape[1]):
        print(f"Fitting marginal KDE for dimension {i}")
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(latent_codes[:, i].reshape(-1, 1))
        marginal_kdes.append(kde)

    # Estimate joint density
    joint_kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(latent_codes)


    # Compute entropy estimates
    def estimate_entropy(kde, samples):
        log_density = kde.score_samples(samples)
        return -np.mean(log_density)


    marginal_entropies = []
    for i in range(latent_codes.shape[1]):
        print(f"Estimating entropy for dimension {i}")
        entropy = estimate_entropy(marginal_kdes[i], latent_codes[:, i].reshape(-1, 1))
        marginal_entropies.append(entropy)

    print(f"Marginal entropies: {marginal_entropies}")
    joint_entropy = estimate_entropy(joint_kde, latent_codes)

    # Calculate Total Correlation
    total_correlation = np.sum(marginal_entropies) - joint_entropy
    print(f'Total Correlation (TC): {total_correlation}')

    print(f"Time taken (seconds): {time.time() - start:.2f}")