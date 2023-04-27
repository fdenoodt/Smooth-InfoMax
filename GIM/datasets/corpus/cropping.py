# %%
import glob
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import split_audio as sa


def crop_by_discarding_back(audio, MAX_LENGTH):
    ''' Crop the audio by discarding the back '''
    return audio[:MAX_LENGTH]


def compute_smallest_length(dir):
    ''' Compute the min length of audio files in a directory '''
    min_length = np.inf
    name = None
    for audio_file in glob.glob(f"{dir}/*.wav"):
        # load via librosa
        audio, sr = librosa.load(audio_file)

        if len(audio) < min_length:
            min_length = len(audio)
            name = audio_file
    return min_length, name


def crop_each_file_in_dir(dir, max_length):
    # for each file in the train and test directories, pad the audio to the max length and save it
    for audio_file in glob.glob(f"{dir}/*.wav"):
        # eg: audio_file = 'split up data/train\\bababa_1_1_ba.wav'
        audio, sr = librosa.load(audio_file)
        audio = crop_by_discarding_back(audio, max_length)

        # remove 'split up data' from the audio_file
        audio_file = audio_file[13:]

        target_path = f"split up data cropped/{audio_file}"
        sf.write(target_path, audio, sr)
        print(len(audio))

        # plt.plot(audio)
        # plt.show()
        # break


if __name__ == '__main__':
    dir_train = 'split up data/train/'
    dir_test = 'split up data/test/'

    min_len_train, name_train = compute_smallest_length(dir_train)
    min_len_test, name_test = compute_smallest_length(dir_test)

    print(f"Min length of train: {min_len_train} ({name_train})")
    print(f"Min length of test: {min_len_test} ({name_test})")

    MIN_LENGTH = min(min_len_train, min_len_test)

    crop_each_file_in_dir(dir_train, MIN_LENGTH)
    crop_each_file_in_dir(dir_test, MIN_LENGTH)

    # %%

    # load gudugi_1_2_du.wav
    # audio, sr = librosa.load('split up data cropped/train/gudugi_1_2_du.wav')
    # plt.plot(audio)
    # plt.show()
