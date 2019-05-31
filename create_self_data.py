from scipy.io import wavfile
from scipy import signal
import numpy as np


def create_data(file_path, bin_size, stride=0):
    fs, data = wavfile.read(file_path)
    data = np.mean(data, axis=1)
    frequencies, times, spectrogram = signal.spectrogram(data, fs)
    after_sec = np.where(times > 1)
    len_example = int(after_sec[0][0] * bin_size)
    if stride == 0:
        stride = len_example
    else:
        stride = int(after_sec[0][0] * stride)
    n_jumps = int(np.floor(len(times)/stride))
    file_samples = []
    for i in range(n_jumps):
        if i*stride+len_example < len(times):
            sample = spectrogram[:,i*stride:i*stride+len_example]
            file_samples.append(sample)
    return np.asarray(file_samples)

