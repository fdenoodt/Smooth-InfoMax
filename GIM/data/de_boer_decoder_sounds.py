from torch.utils.data import Dataset
# from de_boer_sounds import DeBoerDataset

from data import de_boer_sounds

import os
import os.path
import torchaudio
from collections import defaultdict
import torch
import numpy as np
import random



# def load_model(path):
#     print("loading model")
#     # Code comes from: def load_model_and_optimizer()
#     kernel_sizes = [10, 8, 4, 4, 4]
#     strides = [5, 4, 2, 2, 2]
#     padding = [2, 2, 2, 2, 1]
#     enc_hidden = 512
#     reg_hidden = 256

#     calc_accuracy = False
#     num_GPU = None

#     # Initialize model.
#     model = full_model.FullModel(
#         opt,
#         kernel_sizes=kernel_sizes,
#         strides=strides,
#         padding=padding,
#         enc_hidden=enc_hidden,
#         reg_hidden=reg_hidden,
#         calc_accuracy=calc_accuracy,
#     )

#     # Run on only one GPU for supervised losses.
#     if opt["loss"] == 2 or opt["loss"] == 1:
#         num_GPU = 1

#     model, num_GPU = model_utils.distribute_over_GPUs(
#         opt, model, num_GPU=num_GPU)

#     optimizer = torch.optim.Adam(model.parameters(), lr=opt['learning_rate'])
#     model.load_state_dict(torch.load(path, 
#         map_location=device
#         ))

#     return model, optimizer

# def encoder_lambda(xs_batch):
#     # Gim_encoder is outerscope variable
#     with torch.no_grad():
#         return encode(xs_batch, GIM_encoder, depth=1)


# def encode(audio, model, depth=1):
#     audios = audio.unsqueeze(0)
#     model_input = audios.to(device)

#     for idx, layer in enumerate(model.module.fullmodel):
#         context, z = layer.get_latents(model_input)
#         model_input = z.permute(0, 2, 1)

#         if(idx == depth - 1):
#             return z





def default_loader(path):
    return torchaudio.load(path)


def default_flist_reader(flist):
    item_list = []
    speaker_dict = defaultdict(list)
    index = 0
    with open(flist, "r") as rf:
        for line in rf.readlines():
            speaker_id, dir_id, sample_id = line.replace("\n", "").split("-")
            item_list.append((speaker_id, dir_id, sample_id))
            speaker_dict[speaker_id].append(index)
            index += 1

    return item_list, speaker_dict


class DeBoerDecoderDataset(de_boer_sounds.DeBoerDataset):
    def __init__(self, opt, root, directory="train", audio_length=64 * 160, loader=default_loader, encoder=None, background_noise=False, white_guassian_noise=False, target_sample_rate=16000, background_noise_path=None):
        super().__init__(opt, root, directory, audio_length, loader, encoder, background_noise, white_guassian_noise, target_sample_rate, background_noise_path)

        print("XXXXXXXXXXXXXXXXX")

    def __getitem__(self, index):
        


        speaker_id, dir_id, sample_id = self.file_list[index]
        filename = f"{sample_id}"
        audio, samplerate = self.loader(
            os.path.join(self.root, dir_id, f"{filename}.wav")
        )

        assert (
            samplerate == 44100  # todo: before 16000 = 16Khz
        ), "Watch out, samplerate is not consistent throughout the dataset!"

        # resample
        audio = de_boer_sounds.resample(audio,
                              curr_samplerate=44100,
                              new_samplerate=16000)

        # discard last part that is not a full 10ms
        max_length = (audio.size(1) // 160) * 160

        start_idx = random.choice(
            np.arange(160, max_length - self.audio_length - 0, 160)
        )

        # OLD = audio  # audio org: [1, 41278] -> [1, 20480]
        audio = audio[:,  # corresponds to first dim (= 1)
                      start_idx: start_idx + self.audio_length]

        # if encode via GIM is requested
        audio_enc = self.encoder(audio)

        # eg: [1, 2047, 512] -> [2047, 512] -> [512, 2047]
        audio_enc = audio_enc.squeeze(dim=0)
        audio_enc = audio_enc.permute(1, 0)



        return audio, audio_enc, filename, speaker_id, start_idx
