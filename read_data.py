import numpy as np
import os
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt


def get_data(folder_path, save_raw=False):
    all_data = []
    all_y = []
    all_speaker_files = os.listdir(folder_path)

    n_times_per_sec = 71

    for idx, s_f in enumerate(all_speaker_files):
        print(idx)
        speaker_file = folder_path+'\\'+s_f
        chapters = os.listdir(speaker_file)
        for c_f in chapters:
            chapter_file = speaker_file + '\\' + c_f
            raw_samples = os.listdir(chapter_file)
            for s_f in raw_samples:
                sample_file = chapter_file + '\\' + s_f
                if sample_file.endswith('.flac'):
                    data, sample_rate = sf.read(sample_file)
                    n_sub_examples = int(np.floor(len(data) / sample_rate))
                    if save_raw:
                        for i in range(n_sub_examples):
                            all_data.append(data[i * sample_rate:(i + 1) * sample_rate])
                            all_y.append(idx)
                    else:
                        frequencies, times, spectrogram = signal.spectrogram(data, sample_rate)
                        for i in range(n_sub_examples):
                            all_data.append(spectrogram[:,i*n_times_per_sec:(i+1)*n_times_per_sec])
                            all_y.append(idx)

    return np.asarray(all_data), np.asarray(all_y)


def save_train_test():
    train_path = 'C:\\Users\\H\\Desktop\\Hackathon\\LibriSpeech\\dev-clean'
    test_path = 'C:\\Users\\H\\Desktop\\Hackathon\\LibriSpeech\\test-clean'
    x_train, y_train = get_data(train_path)
    x_test, y_test = get_data(test_path)
    np.savez('C:\\Users\\H\\Desktop\\Hackathon\\Data_spec', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


save_train_test()
