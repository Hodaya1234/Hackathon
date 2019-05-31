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


def normalize_data(train_x, train_y, test_x, test_y):
    train_x = 10*np.log10(1+train_x)
    test_x = 10 * np.log10(1 + test_x)
    m = np.mean(train_x, axis=0)
    s = np.std(train_x, axis=0)
    train_x = np.divide(np.subtract(train_x, m), s)
    test_x = np.divide(np.subtract(test_x, m), s)

    [n_train, n_freq, n_times] = train_x.shape

    new_n_freq = int(np.ceil((n_freq - 6) / 4)) * 4 + 6
    new_n_times = int(np.ceil((n_times - 6) / 4)) * 4 + 6

    zero_train = np.zeros([n_train, new_n_freq, new_n_times])
    zero_test = np.zeros([test_x.shape[0], new_n_freq, new_n_times])
    zero_train[:, :n_freq, :n_times] = train_x
    zero_test[:, :n_freq, :n_times] = test_x

    train_x = zero_train[:, np.newaxis, :, :]
    test_x = zero_test[:, np.newaxis, :, :]

    return train_x, train_y, test_x, test_y


def less_data(x, y, new_size=400):
    indices = np.random.choice(len(y), new_size)
    return x[indices, :, :], y[indices]


def save_train_test():
    train_path = 'LibriSpeech\\dev-clean'
    test_path = 'LibriSpeech\\test-clean'
    train_x, train_y = get_data(train_path)
    test_x, test_y = get_data(test_path)

    train_x, train_y = less_data(train_x, train_y, 2000)
    test_x, test_y = less_data(test_x, test_y, 400)

    train_x, train_y, test_x, test_y = normalize_data(train_x, train_y, test_x, test_y)
    np.savez('Data', train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


save_train_test()
