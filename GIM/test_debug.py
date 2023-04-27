# %%
import IPython
import librosa
import torch
import numpy as np

from helper_functions import translate_number_to_syllable

# load
all_audio = torch.load(r"temp_all_audio.pt")
all_cs = torch.load(r"temp_all_cs.pt")
all_labels = torch.load(r"temp_all_labels.pt")

print(all_cs.shape, all_audio.shape, all_labels.shape)

b, c, l = all_audio.shape
# reshuffle all_labels, all_cs, all_audio
idxs = np.arange(b)
np.random.shuffle(idxs)
all_cs = all_cs[idxs]
all_labels = all_labels[idxs]
all_audio = all_audio[idxs]


all_labels = all_labels[:10]
all_cs = all_cs[:10]
all_audio = all_audio[:10]

import IPython.display as ipd

# play all via librosa

for audio, label, c in zip(all_audio, all_labels, all_cs):
    audio = audio[0]
    syllable = translate_number_to_syllable(label)
    print("")
    print(f"label: {label} - {syllable}:")
    ipd.display(ipd.Audio(audio, rate=16000))
    print("c:", c.shape) # (512, )

    # plot c
    import matplotlib.pyplot as plt
    plt.hist(c)
    plt.show()

    # IPython.display.Audio(audio, rate=16000)
