# %%
from decoder.decoder_architectures import *
import torch
from encoder.GIM_encoder import GIM_Encoder
from utils import logger
from data import get_dataloader
import random
from options import OPTIONS as opt

random.seed(0)


def _generate_predictions(decoder, data_loader, encoder, model_nb, path, train_or_test="test"):
    for idx, (batch_org_audio, batch_filenames, _, _) in enumerate(data_loader):
        batch_org_audio = batch_org_audio.to(device)
        batch_per_module = encoder(batch_org_audio)
        batch_enc_audio = batch_per_module[-1].to(device)
        batch_enc_audio = batch_enc_audio.permute(0, 2, 1)  # (b, c, l)

        batch_outp = decoder(batch_enc_audio)

        # to numpy
        batch_org_audio = det_np(batch_org_audio)
        batch_outp = det_np(batch_outp)

        for file_idx, (org_audio, filename, outp) in enumerate(zip(batch_org_audio, batch_filenames, batch_outp)):
            org_audio, outp = org_audio[0], outp[0]  # remove channel dimension

            # frequency domain
            org_mag, outp_mag = fft_magnitude(org_audio), fft_magnitude(outp)

            plot_four_graphs_side_by_side(
                org_audio, outp, org_mag, outp_mag,
                title=f"{filename}, model={model_nb}, True vs Predicted",
                dir=f"{path}/predictions_model={model_nb}/{train_or_test}/",
                file=f"{filename}, model={model_nb}, True vs Predicted", show=False)

            save_audio(outp,
                       f"{path}/predictions_model={model_nb}/{train_or_test}/",
                       file=f"{filename}, model={model_nb}, True vs Predicted", sample_rate=16000)

            if file_idx == 20:
                break  # only do 20 first files
        break  # only do a single batch!


def generate_predictions(options, experiment_name, encoder, criterion, lr, layer_depth, decoder, model_nb=29):
    '''Generate predictions for the test set and save them to disk.'''

    path = f"{options['log_path']}/{criterion}/lr_{lr:.7f}/GIM_L{layer_depth}/"
    model_path = f"{path}/model_{model_nb}.pt"

    decoder.load_state_dict(torch.load(model_path, map_location=device))
    decoder.eval()

    logs = logger.Logger(opt)

    # load the data
    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt, dataset="de_boer_sounds_reshuffledv2", split_and_pad=False, train_noise=False, shuffle=False)

    _generate_predictions(decoder, test_loader, encoder,
                          model_nb, path, train_or_test="test")
    _generate_predictions(decoder, train_loader, encoder,
                          model_nb, path, train_or_test="train")


if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt['batch_size'] = 64 + 32
    opt['batch_size_multiGPU'] = opt['batch_size']

    LAYER_DEPTH = 1
    decoder = SimpleV1Decoder().to(DEVICE)

    CRITERION = "MSE + scMEL Loss Lambda=1.0000000"
    LR = 0.001

    # did best: lr_0.0001/GIM_L{layer_depth}/model_29.pt"
    # works well too lr_1e-05/GIM_L{layer_depth}/model_29.pt"

    # "DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/model_360.ckpt
    # GIM_MODEL_PATH = r"D:\thesis_logs\logs\audio_noise=F_dim=32_distr_wo_nonlin_kld_weight=0.032 !!/model_799.ckpt"
    GIM_MODEL_PATH = r"D:\thesis_logs\logs\de_boer_reshuf_simple_v2_kld_weight=0.0033 !!/model_290.ckpt"

    ENCODER = GIM_Encoder(opt, path=GIM_MODEL_PATH)
    generate_predictions(opt, "GIM_DECODER_experiment", ENCODER, CRITERION, LR,
                         LAYER_DEPTH, decoder, model_nb=29)

    # thought for later: its actually weird i was able to play enc as audio as enc is 512 x something
    # so huh? that means that a lot of info is already in first channel? what do other 511 channels then contain?

    # Observations:
    # First layer decoded still contains the same sound, but with some added noise (could be because decoder hasn't trained very).
    # However, the encoded first layer, still contains the exact sound as the original sound. It is however downsampled a lot -> from 16khz to ~3khz

    # thought for later: its actually weird i was able to play enc as audio as enc is 512 x something
    # so huh? that means that a lot of info is already in first channel? what do other 511 channels then contain?

    # %%
