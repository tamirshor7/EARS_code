#%% imports

import scipy.fftpack
import scipy.io.wavfile as wf
import numpy as np
import os
import matplotlib.pyplot as plt


#%% read recorded signal
data_file_name = 'silent_room'

data_dir_path = os.path.join('..','..','data','real_recordings','13_7_LD_exp_wav')
data_file_path = os.path.join(data_dir_path, data_file_name+'.wav')

fs, data = wf.read(data_file_path)

signal_time_sec = data.shape[0] / fs
x_range = (np.arange(data.shape[0]) / fs)#[int(8*fs):int(19*fs)]


for channel_idx in range(1,5):
    audio_ch = (data.T)[channel_idx]
    # data = data.T[2]
    #data = data[int(8*fs):int(19*fs)]

    #%% check freq spectrum
    # calc fft and show the magnitude spectrum of the signal
    fft_sig = scipy.fftpack.rfft(audio_ch)
    xf = scipy.fftpack.rfftfreq(len(audio_ch), 1 / fs)

    plt.plot(xf, np.abs(fft_sig))

    plt.title(f'Silent room, channel {channel_idx}')
    plt.xlabel('Frequency [Hz]')
    # plt.ylabel('Magnitude [-]')
    plt.xlim(left=0, right=250)
    plt.ylim(top=200, bottom=0)
    plt.show()
    plt.clf()
# %%
