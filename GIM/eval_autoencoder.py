# %%
import importlib
import torch
from options import OPTIONS as opt
import helper_functions
import decoder_architectures
from GIM_encoder import GIM_Encoder
from utils import logger
from data import get_dataloader
import random
from options import OPTIONS as opt

random.seed(0)


if(True):
    importlib.reload(decoder_architectures)
    importlib.reload(helper_functions)

    from decoder_architectures import *
    from helper_functions import *


def generate_predictions(encoder, criterion, lr, layer_depth, decoder, model_nb=29):
    '''Generate predictions for the test set and save them to disk.'''

    path = f"./logs/GIM_DECODER_experiment/{criterion}/lr_{lr}/GIM_L{layer_depth}/"
    model_path = f"{path}/model_{model_nb}.pt"

    decoder.load_state_dict(torch.load(model_path, map_location=device))
    decoder.eval()

    logs = logger.Logger(opt)

    # load the data
    train_loader, _, test_loader, _ = get_dataloader.\
        get_de_boer_sounds_decoder_data_loaders(opt)

    normalize_func = compute_normalizer(train_loader, encoder)

    for idx, (batch_org_audio, batch_filenames, _, _) in enumerate(test_loader):
        batch_org_audio = batch_org_audio.to(device)
        batch_enc_audio = normalize_func(encoder(batch_org_audio).to(device))
        batch_outp = decoder(batch_enc_audio)

        # to numpy
        batch_org_audio = det_np(batch_org_audio)
        batch_outp = det_np(batch_outp)

        for (org_audio, filename, outp) in zip(batch_org_audio, batch_filenames, batch_outp):
            org_audio, outp = org_audio[0], outp[0] # remove channel dimension
            plot_two_graphs_side_by_side(
                org_audio, outp, 
                title=f"{filename}, model={model_nb}, True vs Predicted",
                dir=f"{path}/predictions_model={model_nb}/test/",
                file=f"{filename}, model={model_nb}, True vs Predicted", show=False)
            save_audio(outp, 
                       f"{path}/predictions_model={model_nb}/test/",
                       file=f"{filename}, model={model_nb}, True vs Predicted", sample_rate=16000)
            
        break # only do a single batch!


if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    LAYER_DEPTH = 4
    decoder = GimL4Decoder().to(DEVICE)

    CRITERION = "MSE + Spectral Loss FFT=8192 Lambda=6"
    LR = 0.00001

    # did best: lr_0.0001/GIM_L{layer_depth}/model_29.pt"
    # works well too lr_1e-05/GIM_L{layer_depth}/model_29.pt"

    ENCODER = GIM_Encoder(opt, layer_depth=LAYER_DEPTH,
                          path="DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/model_360.ckpt")
    generate_predictions(ENCODER, CRITERION, LR, LAYER_DEPTH, decoder)

    # thought for later: its actually weird i was able to play enc as audio as enc is 512 x something
    # so huh? that means that a lot of info is already in first channel? what do other 511 channels then contain?

    # Observations:
    # First layer decoded still contains the same sound, but with some added noise (could be because decoder hasn't trained very).
    # However, the encoded first layer, still contains the exact sound as the original sound. It is however downsampled a lot -> from 16khz to ~3khz

    # thought for later: its actually weird i was able to play enc as audio as enc is 512 x something
    # so huh? that means that a lot of info is already in first channel? what do other 511 channels then contain?

    # %%
