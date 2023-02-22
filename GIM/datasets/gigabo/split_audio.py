# %%
# from pydub import AudioSegment

# # Load the audio file
# audio_file = AudioSegment.from_file("path/to/audiofile.wav")

# # Define the segment length (in milliseconds)
# segment_length = 3000

# # Split the audio file into segments of length `segment_length`
# segments = [audio_file[i:i+segment_length] for i in range(0, len(audio_file), segment_length)]

# # Export each segment as a separate file
# for i, segment in enumerate(segments):
#     segment.export(f"output_{i}.wav", format="wav")
import numpy as np
import math
from pyparsing import Iterable
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import split_on_silence

import glob


def plot(y, name, save=None, show=True):
    # plot the waveform
    plt.figure(figsize=(14, 5))
    plt.plot(y)

    # set y-axis scale between 0 and 1
    plt.ylim(-0.6, 0.6)

    # title
    plt.title(name)

    if save:
        # save the plot as a png file
        plt.savefig(f"split up graphs/{save[:-4]}.png")

    if show:
        plt.show()

    plt.clf()
    return None


def plot_three_graphs_side_by_side(y_1, y_2, y_3, name, save=None, show=True):
    ''' Plot three graphs side by side '''

    # plot the waveform
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)  # row, column, index
    plt.plot(y_1)
    plt.ylim(-0.6, 0.6)

    plt.subplot(1, 3, 2)
    plt.plot(y_2)
    plt.ylim(-0.6, 0.6)

    plt.subplot(1, 3, 3)
    plt.plot(y_3)
    plt.ylim(-0.6, 0.6)

    # title for the whole plot
    plt.suptitle(name)

    if save:
        # save the plot as a png file
        plt.savefig(f"split up graphs/{save[:-4]}_split.png")

    if show:
        plt.show()

    plt.clf()
    return None


def chunks(mask, reference_indices, file):
    ''' Compute indices used for splitting into chunks '''
    n = len(reference_indices)

    indices = []
    for idx, current in enumerate(mask):
        left = mask[idx - 1]
        if left == 1 and current == 0:
            indices.append(idx)

    # remove last index
    indices = indices[:-1]  # last split point doesn't count

    # assert len(indices) == n - 1 # 3 chunks -> 2 split points

    # ---- x  ----  x ----
    # -- x  -- x - x ---

    if(len(indices)) != n:
        print(f"Error: len(indices) != n for {file}")
        indices_filtered = []
        for idx, reference_value in enumerate(reference_indices):
            # select the 3 indices closest to the reference indices
            indices_filtered.append(
                min(indices, key=lambda x: abs(x - reference_value)))

        # assert indices_filtered is unique
        assert len(indices_filtered) == len(set(indices_filtered))
        return indices_filtered
    else:
        return indices


def threshold_otsu(x: Iterable, *args, **kwargs) -> float:
    """Find the threshold value for a bimodal histogram using the Otsu method.
    https://stackoverflow.com/questions/48213278/implementing-otsu-binarization-from-scratch-python

    If you have a distribution that is bimodal (AKA with two peaks, with a valley
    between them), then you can use this to find the location of that valley, that
    splits the distribution into two.
    

    From the SciKit Image threshold_otsu implementation:
    https://github.com/scikit-image/scikit-image/blob/70fa904eee9ef370c824427798302551df57afa1/skimage/filters/thresholding.py#L312
    """
    counts, bin_edges = np.histogram(x, *args, **kwargs)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold


def main(dir):
    ctr = 1
    for file in glob.glob(f"{dir}/*.wav"):
        if(ctr == 20):
            break

        ctr += 1

        # eg test\\babibu_1.wav'
        name = file[:-4]  # remove .wav
        name = name.split("\\")[1]  # remove test\\
        # print(name)

        # if not name == "babada_1":
        #     continue

        # load wav file using librosa
        y, sr = librosa.load(file)

        segment_length = math.ceil(len(y) / 3)
        reference_indices = [segment_length, segment_length * 2]

        y_abs = np.abs(y)

        # sliding window over signal
        # if sr = 22050, window size is 500 = 0.022675736961451247 seconds
        WINDOW_SIZE = int(sr // 44)
        y_max_sliding_window = np.array([np.max(y_abs[i:i+WINDOW_SIZE]) for i in range(len(y_abs)-WINDOW_SIZE)])
        y_mean_sliding_window = np.array([np.percentile(y_abs[i:i+WINDOW_SIZE], 90) for i in range(len(y_abs)-WINDOW_SIZE)])

        # create mask for large jumps
        treshold = threshold_otsu(y_mean_sliding_window)
        print(f"th: {treshold}")

        # treshold = 0.2
        mask = np.where(y_max_sliding_window > treshold, 1, 0)
        try:
            indices = chunks(mask, reference_indices, file)
        except:
            print("****************")
            print(f"Error: {file}")
            continue

        # plot(mask, name="idk")
        # plot(y_max_sliding_window, name="idk")
        # plot(y_mean_sliding_window, name="idk")


        y_1 = y[0:indices[0]]
        y_2 = y[indices[0]:indices[1]]
        y_3 = y[indices[1]:]


        plot(y, name=f"Soundwave {name}", save=file, show=False)
        plot_three_graphs_side_by_side(
            y_1, y_2, y_3, name=f"Soundwave - split up {name}", save=file, show=False)

        # save using soundfile
        sf.write(f"split up data/{file[:-4]}_1.wav", y_1, sr)
        sf.write(f"split up data/{file[:-4]}_2.wav", y_2, sr)
        sf.write(f"split up data/{file[:-4]}_3.wav", y_3, sr)


if __name__ == "__main__":


    print("Splitting up audio files - train")
    main("train")

    print("Splitting up audio files - test")
    main("test")

# %%
