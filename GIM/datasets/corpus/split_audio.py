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

# matplotlib theme


plt.rcParams['axes.titlesize'] = 18
plt.rcParams['figure.titlesize'] = 18





def plot(y, name, save=None, show=True):
    # plot the waveform
    plt.figure(figsize=(14, 5))
    plt.plot(y)

    # set y-axis scale between 0 and 1
    # plt.ylim(-0.6, 0.6)

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

import seaborn as sns

def plot_histogram(histogram, x_bins, name, threshold=None):
    ''' Plot a histogram '''

    colors = sns.color_palette('muted')


    # plot the histogram
    plt.figure(figsize=(8, 5))
    # plt.bar(x_bins, histogram, width=0.01, color = colors[1])
    plt.bar(x_bins, histogram, width=0.01)
    plt.xlim(min(x_bins)-0.01, max(x_bins) + 0.01)
    
    # title
    plt.title(name)

    if threshold:
        plt.axvline(x=threshold, color='r', linestyle='--')

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

    # print(threshold)

    # plot_histogram(counts, bin_centers, "Histogram bababi_1", threshold)

    return threshold


def main(dir):
    ctr = 1
    for file in glob.glob(f"{dir}/*.wav"):
        # if(ctr == 2):
        #     break

        ctr += 1

        # eg test\\babibu_1.wav'
        name = file[:-4]  # remove .wav
        name = name.split("\\")[1]  # remove test\\
        # print(name)

        # if not name == "babada_1":
        # if not name == "bababi_1":
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
        percentile = 90
        y_nth_percentile_sliding_window = np.array([np.percentile(y_abs[i:i+WINDOW_SIZE], percentile) for i in range(len(y_abs)-WINDOW_SIZE)])

        # create mask for large jumps
        treshold = threshold_otsu(y_nth_percentile_sliding_window)
        # print(f"th: {treshold}")

        # treshold = 0.2
        mask = np.where(y_max_sliding_window > treshold, 1, 0)
        try:
            indices = chunks(mask, reference_indices, file)
        except:
            print("****************")
            print(f"Error: {file}")
            continue

        # plot(y_max_sliding_window, name=f"Soundwave {name} - max sliding window")
        # plot(y_nth_percentile_sliding_window, name=f"Soundwave {name} - {percentile}th percentile sliding window")
        # plot(mask, name=f"Soundwave {name} - mask")


        y_1 = y[0:indices[0]]
        y_2 = y[indices[0]:indices[1]]
        y_3 = y[indices[1]:]


        plot(y, name=f"Soundwave {name}", save=file, show=False)
        plot_three_graphs_side_by_side(
            y_1, y_2, y_3, name=f"Soundwave - split up {name}", save=file, show=False)
        
        # name eg babibu_1
        # split up into "ba", "bi", "bu"
        name1 = name[0:2]
        name2 = name[2:4]
        name3 = name[4:6]

        # save using soundfile
        sf.write(f"split up data/{file[:-4]}_1_{name1}.wav", y_1, sr)
        sf.write(f"split up data/{file[:-4]}_2_{name2}.wav", y_2, sr)
        sf.write(f"split up data/{file[:-4]}_3_{name3}.wav", y_3, sr)


if __name__ == "__main__":


    print("Splitting up audio files - train")
    main("train")

    print("Splitting up audio files - test")
    main("test")

# %%




# to deal with:
# Error: len(indices) != n for train\babidi_1.wav
# Error: len(indices) != n for train\babigi_1.wav
# Error: len(indices) != n for train\babigu_1.wav
# Error: len(indices) != n for train\babuba_1.wav
# Error: len(indices) != n for train\badabu_1.wav
# Error: len(indices) != n for train\badada_1.wav
# Error: len(indices) != n for train\badiba_1.wav
# Error: len(indices) != n for train\badida_1.wav
# Error: len(indices) != n for train\baduda_1.wav
# Error: len(indices) != n for train\baduga_1.wav
# Error: len(indices) != n for train\bagagi_1.wav
# Error: len(indices) != n for train\bagiba_1.wav
# Error: len(indices) != n for train\bagibi_1.wav
# Error: len(indices) != n for train\bagibu_1.wav
# Error: len(indices) != n for train\bagiga_1.wav
# Error: len(indices) != n for train\bagigi_1.wav
# Error: len(indices) != n for train\bagigu_1.wav
# ****************
# Error: train\bagigu_1.wav
# Error: len(indices) != n for train\bibadi_1.wav
# Error: len(indices) != n for train\bibagi_1.wav
# Error: len(indices) != n for train\bibiba_1.wav
# Error: len(indices) != n for train\bibidi_1.wav
# Error: len(indices) != n for train\bibuba_1.wav
# Error: len(indices) != n for train\bibudi_1.wav
# Error: len(indices) != n for train\bibudu_1.wav
# Error: len(indices) != n for train\bidaba_1.wav
# Error: len(indices) != n for train\bidabi_1.wav
# Error: len(indices) != n for train\bidagi_1.wav
# Error: len(indices) != n for train\bidibi_1.wav
# Error: len(indices) != n for train\bidubi_1.wav
# Error: len(indices) != n for train\biduga_1.wav
# Error: len(indices) != n for train\bigaba_1.wav
# Error: len(indices) != n for train\bigaga_1.wav
# Error: len(indices) != n for train\bigagi_1.wav
# Error: len(indices) != n for train\bigigi_1.wav
# Error: len(indices) != n for train\bigigu_1.wav
# Error: len(indices) != n for train\biguda_1.wav
# Error: len(indices) != n for train\bigudi_1.wav
# Error: len(indices) != n for train\bigudu_1.wav
# Error: len(indices) != n for train\bigugi_1.wav
# Error: len(indices) != n for train\bubidi_1.wav
# Error: len(indices) != n for train\bubidu_1.wav
# Error: len(indices) != n for train\bubigi_1.wav
# Error: len(indices) != n for train\budadi_1.wav
# Error: len(indices) != n for train\budaga_1.wav
# Error: len(indices) != n for train\budidi_1.wav
# Error: len(indices) != n for train\bududa_1.wav
# Error: len(indices) != n for train\bududu_1.wav
# Error: len(indices) != n for train\bugibi_1.wav
# Error: len(indices) != n for train\bugida_1.wav
# Error: len(indices) != n for train\bugidi_1.wav
# Error: len(indices) != n for train\bugudi_1.wav
# Error: len(indices) != n for train\dabadi_1.wav
# Error: len(indices) != n for train\dabidi_1.wav
# Error: len(indices) != n for train\dabuda_1.wav
# Error: len(indices) != n for train\dabugi_1.wav
# Error: len(indices) != n for train\dadabi_1.wav
# Error: len(indices) != n for train\dadada_1.wav
# Error: len(indices) != n for train\dadagu_1.wav
# Error: len(indices) != n for train\dadiba_1.wav
# Error: len(indices) != n for train\dadida_1.wav
# Error: len(indices) != n for train\dadidu_1.wav
# Error: len(indices) != n for train\dadiga_1.wav
# Error: len(indices) != n for train\dadubu_1.wav
# Error: len(indices) != n for train\dadugi_1.wav
# Error: len(indices) != n for train\dagiba_1.wav
# ****************
# Error: train\dagiba_1.wav
# Error: len(indices) != n for train\dagibu_1.wav
# Error: len(indices) != n for train\dagigu_1.wav
# Error: len(indices) != n for train\dagubi_1.wav
# Error: len(indices) != n for train\dibabi_1.wav
# Error: len(indices) != n for train\dibada_1.wav
# Error: len(indices) != n for train\dibadi_1.wav
# Error: len(indices) != n for train\dibagi_1.wav
# Error: len(indices) != n for train\dibibi_1.wav
# Error: len(indices) != n for train\dibida_1.wav
# Error: len(indices) != n for train\dibiga_1.wav
# Error: len(indices) != n for train\dibigi_1.wav
# Error: len(indices) != n for train\dibigu_1.wav
# Error: len(indices) != n for train\dibuba_1.wav
# Error: len(indices) != n for train\dibudi_1.wav
# Error: len(indices) != n for train\dibudu_1.wav
# Error: len(indices) != n for train\dibugi_1.wav
# Error: len(indices) != n for train\didabi_1.wav
# Error: len(indices) != n for train\didaga_1.wav
# Error: len(indices) != n for train\didagi_1.wav
# Error: len(indices) != n for train\didagu_1.wav
# Error: len(indices) != n for train\dididi_1.wav
# Error: len(indices) != n for train\didigi_1.wav
# Error: len(indices) != n for train\didigu_1.wav
# Error: len(indices) != n for train\didubi_1.wav
# Error: len(indices) != n for train\diduga_1.wav
# Error: len(indices) != n for train\didugi_1.wav
# Error: len(indices) != n for train\didugu_1.wav
# Error: len(indices) != n for train\digaba_1.wav
# Error: len(indices) != n for train\digabi_1.wav
# Error: len(indices) != n for train\digada_1.wav
# Error: len(indices) != n for train\digadi_1.wav
# Error: len(indices) != n for train\digagu_1.wav
# Error: len(indices) != n for train\digiba_1.wav
# Error: len(indices) != n for train\digidu_1.wav
# Error: len(indices) != n for train\digiga_1.wav
# Error: len(indices) != n for train\diguba_1.wav
# Error: len(indices) != n for train\dubaba_1.wav
# Error: len(indices) != n for train\dubada_1.wav
# Error: len(indices) != n for train\dubagi_1.wav
# Error: len(indices) != n for train\dubigi_1.wav
# Error: len(indices) != n for train\dudabu_1.wav
# Error: len(indices) != n for train\dudagi_1.wav
# Error: len(indices) != n for train\dudagu_1.wav
# Error: len(indices) != n for train\dudigu_1.wav
# Error: len(indices) != n for train\duduga_1.wav
# Error: len(indices) != n for train\dugadi_1.wav
# Error: len(indices) != n for train\dugaga_1.wav
# Error: len(indices) != n for train\dugida_1.wav
# Error: len(indices) != n for train\dugidu_1.wav
# Error: len(indices) != n for train\dugiga_1.wav
# Error: len(indices) != n for train\dugigi_1.wav
# Error: len(indices) != n for train\dugubi_1.wav
# Error: len(indices) != n for train\dugugi_1.wav
# Error: len(indices) != n for train\gabigi_1.wav
# Error: len(indices) != n for train\gabudu_1.wav
# Error: len(indices) != n for train\gabugi_1.wav
# Error: len(indices) != n for train\gadaba_1.wav
# Error: len(indices) != n for train\gadadu_1.wav
# Error: len(indices) != n for train\gadagi_1.wav
# Error: len(indices) != n for train\gadibu_1.wav
# Error: len(indices) != n for train\gadida_1.wav
# Error: len(indices) != n for train\gadidu_1.wav
# Error: len(indices) != n for train\gadiga_1.wav
# Error: len(indices) != n for train\gadigu_1.wav
# Error: len(indices) != n for train\gaduba_1.wav
# Error: len(indices) != n for train\gadubu_1.wav
# Error: len(indices) != n for train\gadudu_1.wav
# Error: len(indices) != n for train\gagagi_1.wav
# Error: len(indices) != n for train\gagidu_1.wav
# Error: len(indices) != n for train\gagiga_1.wav
# Error: len(indices) != n for train\gibaba_1.wav
# Error: len(indices) != n for train\gibabu_1.wav
# Error: len(indices) != n for train\gibada_1.wav
# Error: len(indices) != n for train\gibaga_1.wav
# Error: len(indices) != n for train\gibagi_1.wav
# Error: len(indices) != n for train\gibagu_1.wav
# Error: len(indices) != n for train\gibiba_1.wav
# Error: len(indices) != n for train\gibibu_1.wav
# Error: len(indices) != n for train\gibida_1.wav
# Error: len(indices) != n for train\gibidu_1.wav
# Error: len(indices) != n for train\gibiga_1.wav
# Error: len(indices) != n for train\gibigu_1.wav
# Error: len(indices) != n for train\gibubi_1.wav
# Error: len(indices) != n for train\gibubu_1.wav
# Error: len(indices) != n for train\gibuda_1.wav
# Error: len(indices) != n for train\gibudi_1.wav
# Error: len(indices) != n for train\gibugu_1.wav
# Error: len(indices) != n for train\gidabi_1.wav
# Error: len(indices) != n for train\gidada_1.wav
# Error: len(indices) != n for train\gidadu_1.wav
# Error: len(indices) != n for train\gidagi_1.wav
# Error: len(indices) != n for train\gidagu_1.wav
# Error: len(indices) != n for train\gidibu_1.wav
# Error: len(indices) != n for train\gidigi_1.wav
# Error: len(indices) != n for train\gidigu_1.wav
# Error: len(indices) != n for train\giduba_1.wav
# Error: len(indices) != n for train\gidubi_1.wav
# Error: len(indices) != n for train\gidubu_1.wav
# Error: len(indices) != n for train\giduga_1.wav
# Error: len(indices) != n for train\gidugi_1.wav
# Error: len(indices) != n for train\gidugu_1.wav
# Error: len(indices) != n for train\gigabi_1.wav
# Error: len(indices) != n for train\gigadi_1.wav
# Error: len(indices) != n for train\gigadu_1.wav
# Error: len(indices) != n for train\gigiba_1.wav
# Error: len(indices) != n for train\gigibi_1.wav
# Error: len(indices) != n for train\gigidi_1.wav
# Error: len(indices) != n for train\gigidu_1.wav
# Error: len(indices) != n for train\gigiga_1.wav
# Error: len(indices) != n for train\gigigu_1.wav
# Error: len(indices) != n for train\gigubu_1.wav
# Error: len(indices) != n for train\giguda_1.wav
# Error: len(indices) != n for train\giguga_1.wav
# Error: len(indices) != n for train\gigugi_1.wav
# Error: len(indices) != n for train\gubadu_1.wav
# Error: len(indices) != n for train\gubiba_1.wav
# Error: len(indices) != n for train\gudadi_1.wav
# Error: len(indices) != n for train\gudagi_1.wav
# Error: len(indices) != n for train\gudigi_1.wav
# Error: len(indices) != n for train\gududi_1.wav
# Error: len(indices) != n for train\gududu_1.wav
# Error: len(indices) != n for train\gugabu_1.wav
# Error: len(indices) != n for train\gugadi_1.wav
# Error: len(indices) != n for train\gugagi_1.wav
# Error: len(indices) != n for train\gugibi_1.wav
# Error: len(indices) != n for train\gugidi_1.wav
# Error: len(indices) != n for train\gugiga_1.wav
# Error: len(indices) != n for train\gugudu_1.wav
# Error: len(indices) != n for train\gugugi_1.wav
# Splitting up audio files - test
# Error: len(indices) != n for test\bigagi_1.wav
# Error: len(indices) != n for test\bubaga_1.wav
# Error: len(indices) != n for test\bubida_1.wav
# Error: len(indices) != n for test\gabigi_1.wav
# Error: len(indices) != n for test\gadiga_1.wav
# Error: len(indices) != n for test\gagabi_1.wav
# Error: len(indices) != n for test\gibagi_1.wav
# Error: len(indices) != n for test\gibubu_1.wav
# Error: len(indices) != n for test\gidabu_1.wav
# Error: len(indices) != n for test\gidiga_1.wav
# Error: len(indices) != n for test\gidubu_1.wav
# Error: len(indices) != n for test\gigiba_1.wav
# Error: len(indices) != n for test\gigigi_1.wav
# Error: len(indices) != n for test\gugagi_1.wav
