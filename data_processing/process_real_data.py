'''
Process a recordings session and plot:
1. A part of all signals and rotor's angular position, while compensating the phase of mics.
2. Magnitude spectrum
3. Avg audio channels - averaging by the angular position of the rotor (not time)
4. Audio peaks VS mics angular coordinates
'''
#%%
# imports
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from scipy.signal import hilbert

# import local methods
from pre_processing import *
from utils import read_recordings_from_h5

#%% 
# read data from h5
data_exp_name = '13_9_75'
fs = 44100
rotor_type = 18
mode = 'CCW' #'CCW' or 'CW' for counter-clockwise or clockwise, respectively.
vel_rate = 5

audio_channels, rotor_angle, mics_data, rps, bpf, _, mics_distances_sorted, mics_distances_sorted_idx = read_recordings_from_h5(data_exp_name, fs, rotor_type, mode, vel_rate)

chosen_audio_channels = [0,1,2,3]
colors = ['b','r','g','orange']

#%%
# plot audio channels and encoder
audio_samples = []

rotor_angle_plot_scale = 0.00001
partial_plot_scale = 0.3
end = int(fs*partial_plot_scale)
start = int(fs*0.1)
for idx_loop, channel_id_loop in enumerate(chosen_audio_channels):
    sample_filtered = butter_lowpass_filter(audio_channels[channel_id_loop], 500, 44100, order=2)
    #sample_filtered = butter_lowpass_filter(sample_filtered, rps, 44100, order=2, by='highpass')

    audio_samples.append(sample_filtered)

    # roll channels by TD of the angle between mics
    dist_mic = mics_data.loc[channel_id_loop, 'CW angular position [deg]']
    audio_samples[idx_loop] = np.roll(audio_samples[idx_loop], int((fs * dist_mic) / (np.pi * bpf)))

    x_range = np.arange(audio_samples[idx_loop].shape[0]) / fs
    plt.plot(x_range[start:end], audio_samples[idx_loop][start:end], label=f'Audio channel {channel_id_loop+1}', alpha=0.7, 
            color=colors[channel_id_loop])

plt.plot(x_range[start:end], rotor_angle[start:end] * rotor_angle_plot_scale, label='Rotor angle', c='black', linewidth=1)
plt.legend(loc='lower left')
plt.xlabel('Time [sec]')
plt.title(f'Audio Channels by Rotor Angle, rotor {rotor_type}, RPS={round(rps,2)}')
plt.show()
plt.clf()

#%% 
# magnitude spectrum
right_lim = bpf * 5

for idx_loop, channel_id_loop in enumerate(mics_distances_sorted_idx):
    audio_sample = audio_samples[channel_id_loop]

    fft_sig = scipy.fftpack.rfft(audio_sample)
    xf = scipy.fftpack.rfftfreq(len(audio_sample), 1 / fs)
    plt.plot(xf, np.abs(fft_sig),label=f'Channel {channel_id_loop+1}', color=colors[channel_id_loop])
    plt.xlim(left=0, right=right_lim)
    if idx_loop == len(chosen_audio_channels) - 1:        
        xs = np.arange(bpf, bpf * 5, bpf)
        ys = np.ones_like(xs)
        plt.scatter(xs, ys, facecolors='none', s=80, edgecolors='green', label='BPF harmonics', zorder=3)
        # rps
        plt.scatter(int(rps),np.ones_like(rps), facecolors='none', s=80, edgecolors='red', label='RPS', zorder=3)

    # plt.ylim(top=70)
plt.legend()
plt.title(f'Magnitude spectrum, rotor {rotor_type}, RPS={round(rps,2)}')
plt.xlabel('Frequency [Hz]')
plt.show()
plt.clf()
#%% 
# create avg audio for the four mics

avg_audio_all = []
rotor_audio_all = []
rotor_angle_step = 8

for channel_id_loop in chosen_audio_channels:
    avg_audio_single_channel = []
    audio_sample = audio_samples[channel_id_loop]

    for i in range(min(rotor_angle), max(rotor_angle), rotor_angle_step):
        avg_audio_single_channel.append([i, np.mean(audio_sample[rotor_angle == i])])

    avg_audio = np.array(avg_audio_single_channel)
    x_range =  2 * np.pi * avg_audio[:,0] / max(rotor_angle)
    avg_audio = avg_audio[:,1]

    plt.plot(x_range, avg_audio, label=f'Channel {channel_id_loop+1}', color=colors[channel_id_loop])
    
    avg_audio_all.append(avg_audio)

plt.title(f'Averaged Audio Channels, rotor {rotor_type}, RPS={round(rps,2)}')
plt.legend()
plt.grid()
plt.xlabel('Rotor angle [rad]')
plt.show()
plt.clf()
#%% 
# peaks vs radial coord

def func(x, a, c):
    return a / x**2 + c

avg_audio_all_np = np.array(avg_audio_all)

maxes = np.max(avg_audio_all_np, 1)[mics_distances_sorted_idx]
mins = np.min(avg_audio_all_np, 1)[mics_distances_sorted_idx]

popt, pcov = curve_fit(func, mics_distances_sorted, maxes)#, bounds=([0,0], [np.inf, np.inf]))

x_range = np.linspace(min(mics_distances_sorted)-5, max(mics_distances_sorted)+5, 500)
plt.plot(x_range, func(x_range, *popt), 'r-', label='fit: a=%5.3f, c=%5.3f' % tuple(popt))

plt.plot(mics_distances_sorted, maxes, label='max peak', marker='o')

plt.title(f'Audio peaks VS mics angular coordinates, rotor {rotor_type}, RPS={round(rps,2)}')
plt.xlabel('Mic distance from rotor\'s origin[cm]')
plt.ylabel('Audio amplitude')
plt.grid()
plt.legend()
plt.show()
plt.clf()

#%%
# hilbert transform
idx_channel = 3
partial_sig = 0.2
signal = audio_samples[idx_channel][:int(audio_samples[idx_channel].shape[0]*partial_sig)]
freq = rps
signal = butter_bandpass_filter(signal, lowcut=freq*0.9, highcut=freq*1.1, fs=fs, order=2)
sig_shape = signal.shape[0]
signal = signal[int(sig_shape * 0.1): int(sig_shape * (1-0.1))]

analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
x_range = np.arange(signal.shape[0]) / fs
plt.plot(x_range, signal, label='signal')
plt.plot(x_range, amplitude_envelope, label='envelope')
plt.show()
plt.clf()

# %%
