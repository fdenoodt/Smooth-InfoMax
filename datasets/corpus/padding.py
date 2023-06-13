# %%
import glob
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import split_audio as sa


def pad_audio_front_and_back(audio, MAX_LENGTH):
    ''' Pad the audio to the max length by adding zeros to the front and back '''
    # compute the number of zeros to pad by
    num_zeros = MAX_LENGTH - len(audio)

    # compute the number of zeros to pad on the front and back
    num_zeros_front = num_zeros // 2
    num_zeros_back = num_zeros - num_zeros_front

    # pad the audio
    audio = np.pad(audio, (num_zeros_front, num_zeros_back), 'constant', constant_values=0)
    return audio


# compute the number of samples to pad by iterating through the audio files and finding the max length

def compute_padding_length(dir):
    ''' Compute the max length of audio files in a directory '''
    max_length = 0
    name = None
    for audio_file in glob.glob(f"{dir}/*.wav"):
           # load via librosa
        audio, sr = librosa.load(audio_file)

        if len(audio) > max_length:
            max_length = len(audio)
            name = audio_file
    return max_length, name

def add_padding_to_each_file_in_dir(dir, max_length):
    ''' Add padding to each audio file in a directory'''
    # for each file in the train and test directories, pad the audio to the max length and save it
    for audio_file in glob.glob(f"{dir}/*.wav"):
        # eg: audio_file = 'split up data/train\\bababa_1_1_ba.wav'
        audio, sr = librosa.load(audio_file)
        audio = pad_audio_front_and_back(audio, max_length)

        
        #remove 'split up data' from the audio_file
        audio_file = audio_file[13:]

        target_path = f"split up data padded/{audio_file}"
        sf.write(target_path, audio, sr)

        plt.plot(audio)
        plt.show()
        break


if __name__ == '__main__':
    dir_train = 'split up data/train/'
    dir_test = 'split up data/test/'
    
    max_len_train, name_train = compute_padding_length(dir_train)
    max_len_test, name_test = compute_padding_length(dir_test)

    print(f"Max length of train: {max_len_train} ({name_train})")
    print(f"Max length of test: {max_len_test} ({name_test})")

    MAX_LENGTH = max(max_len_train, max_len_test)

    add_padding_to_each_file_in_dir(dir_train, MAX_LENGTH)
    add_padding_to_each_file_in_dir(dir_test, MAX_LENGTH)


    # %%

    # load gudugi_1_2_du.wav
    audio, sr = librosa.load('split up data padded/train/gudugi_1_2_du.wav')
    plt.plot(audio)
    plt.show()

