import torchaudio
import numpy as np
import random
import torchaudio
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback


def default_loader(path):
    return torchaudio.load(path, normalize=False)


# mean = -1456218.7500
# std = 135303504.0
audio_length = 20480

path = "C:\\GitHub\\Smooth-InfoMax\\datasets\\LibriSpeech\\train-clean-100\\26\\496\\26-496-0000.flac"
audio, samplerate = default_loader(path)

assert (
        samplerate == 16000
), "Watch out, samplerate is not consistent throughout the dataset!"

# discard last part that is not a full 10ms
max_length = audio.size(1) // 160 * 160

start_idx = random.choice(
    np.arange(160, max_length - audio_length - 0, 160)
)

audio = audio[:, start_idx: start_idx + audio_length]

# Calculate the mean and std based on the actual audio data
# mean = audio.mean()
# std = audio.std()
#
#
# # normalize
# audio = (audio - mean) / std
#
# # save to wav
# # unnormalize
# audio = audio * std + mean
audio = audio.float()
torchaudio.save("x.wav", audio, 16000)

audio = audio.squeeze(0).cpu().numpy()


class LogAudioCallback(Callback):
    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger
        self.wandb_logger.experiment.log({"audioa": wandb.Audio(audio, sample_rate=16000)})


wandb_logger = WandbLogger(project="test")
trainer = Trainer(logger=wandb_logger, callbacks=[LogAudioCallback(wandb_logger)])
# Assuming you have a LightningModule `model`
# trainer.fit(model)

wandb.finish()
