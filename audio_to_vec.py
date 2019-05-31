import sys
import numpy as np
import os
import soundfile as sf
from scipy import signal
from CNN import ConvNet
import torch
from torch import nn


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def create_testing():
    len_example = 71
    sample_rate = 16000
    speaker_folders = os.listdir('testing')
    all_examples = []
    all_labels = []
    for idx, speaker in enumerate(speaker_folders):
        speaker_rawdata = np.asarray([0])
        files = os.listdir('testing' + '\\' + speaker)
        for file in files:
            data, _ = sf.read('testing' + '\\' + speaker + '\\' + file)
            speaker_rawdata = np.concatenate([speaker_rawdata, data])
        speaker_rawdata = speaker_rawdata[1:]
        frequencies, times, spectrogram = signal.spectrogram(speaker_rawdata, sample_rate)
        n_sub_examples = int(np.floor(len(times)/len_example))
        speaker_examples = []
        for i in range(n_sub_examples):
            all_examples.append(spectrogram[:, i * len_example:(i + 1) * len_example])
            all_labels.append(idx)
    np.save('testing_spec', np.asarray(all_examples))
    np.save('testing_labels', np.asarray(all_labels))


def run():
    samples = np.load('testing_spec.npy')
    model = ConvNet(130, 74, 40)
    model.load_state_dict(torch.load('best_model.pth'))
    model.fc2 = Identity()
    model = model.double()
    samples = np.asarray(samples)

    [n_train, n_freq, n_times] = samples.shape

    new_n_freq = int(np.ceil((n_freq - 6) / 4)) * 4 + 6
    new_n_times = int(np.ceil((n_times - 6) / 4)) * 4 + 6

    zero_train = np.zeros([n_train, new_n_freq, new_n_times])
    zero_train[:, :n_freq, :n_times] = samples

    samples = zero_train[:, np.newaxis, :, :]
    vecs = []
    with torch.no_grad():
        for s in samples:
            s = s[np.newaxis,:,:,:]
            res = model(torch.from_numpy(s))
            vecs.append(res.numpy())

    np.save('testing_vectors', np.asarray(vecs))
# take the entire movie, divide it to one second parts and cluster



run()