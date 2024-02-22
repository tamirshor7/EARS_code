#%% imports
import numpy as np
import os, sys

# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import pandas as pd

import scipy
from scipy.optimize import curve_fit

from utils import *
from pre_processing import *
cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from forward_model.RealRecordings import RealRecordings

#%% params
data_dir_name_prefix = '13_9_'
rotor_type = 18
mode = 'CCW'
vel_rate = 14

const_diff_mic = 17
radiuses = [str(x) for x in sorted(list(range(70,125,10)) + [75, 85])]
vel_rates = np.arange(15)
num_audio_channels = 4

#%% create classes instances
real_recordings_per_vel = []
for radius in radiuses:
    data_exp_name = data_dir_name_prefix + radius
    data_dir_path = os.path.join('..','..','data','real_recordings',data_exp_name)
    real_recordings_per_radius = RealRecordings(data_dir_path, recordings_mode=mode, rotor_type=rotor_type, 
                                                vel_rate=vel_rate, compansate_on_mics_phase=True,
                                                low_cut_coeff=0.9, high_cut_coeff=5.1)
    real_recordings_per_vel.append(real_recordings_per_radius)
# %%
for idx_radius, radius in enumerate(radiuses):
    cur_real_recordings = real_recordings_per_vel[idx_radius]
    for audio_channel in range(num_audio_channels):
        y = cur_real_recordings.avg_audio_channels[audio_channel]
        plt.plot(cur_real_recordings.encoder_readings_for_avg_audio, y, label=f'Channel {audio_channel}')

    plt.xlabel('Rotor Angle [rad]')
    plt.title(f'Averaged Recordings (compansated over mics phases)\nRadius {int(radius)-const_diff_mic}[cm], Rotor {rotor_type} {mode}, RPS {round(cur_real_recordings.rps,2)}[Hz]')
    plt.legend(loc='upper left')
    plt.show()
    plt.clf()


    # calc fft and show the magnitude spectrum of the signal

    fig, axes = plt.subplots(4, sharex=True, sharey=True)

    for audio_channel in range(num_audio_channels):
        y2 = cur_real_recordings.audio_channels[audio_channel]

        fft_sig = scipy.fftpack.rfft(y2)
        xf = scipy.fftpack.rfftfreq(len(y2), 1 / cur_real_recordings.fs)

        bpf_harmonics = np.arange(cur_real_recordings.rps*2, 
                                cur_real_recordings.rps*10, 
                                cur_real_recordings.rps*2)
        
        axes[audio_channel].plot(xf, np.abs(fft_sig), zorder=1)
        axes[audio_channel].scatter(bpf_harmonics, (np.abs(fft_sig))[8*bpf_harmonics.astype(np.int)], 
                    s=80, facecolors='none', edgecolors='r', 
                    label='BPF harmonics', zorder=2) #(np.abs(fft_sig))[bpf_harmonics.astype(np.int)], 
        axes[audio_channel].scatter(cur_real_recordings.rps, np.abs(fft_sig)[8*int(cur_real_recordings.rps)], 
                    s=80, facecolors='none', edgecolors='green', 
                    label='RPS', zorder=2)
    fig.supxlabel('Frequency [Hz]')
    fig.supylabel('Magnitude [-]')
    plt.xlim(left=0, right=300)
    plt.legend(loc='lower right')
    fig.suptitle(f'Magnitude Spectrum\nRadius {int(radius)-const_diff_mic}[cm], Rotor {rotor_type} {mode}, RPS {round(cur_real_recordings.rps,2)}[Hz]')
    plt.show()
    plt.clf()
# %%
for audio_channel in range(num_audio_channels):
    
    for idx_radius, radius in enumerate(radiuses):
        cur_real_recordings = real_recordings_per_vel[idx_radius]

        y = cur_real_recordings.avg_audio_channels[audio_channel]
        plt.plot(cur_real_recordings.encoder_readings_for_avg_audio, y, label=f'Radius {int(radius)-const_diff_mic}[cm]')

    plt.xlabel('Rotor Angle [rad]')
    plt.title(f'Averaged Recordings (compansated over mics phases)\nChannel {audio_channel}, Rotor {rotor_type} {mode}, RPS {round(cur_real_recordings.rps,2)}[Hz]')
    plt.legend(loc='upper left')
    plt.show()
    plt.clf()


# #%%
# avg_all_channels_all_radiuses = []
# for idx_radius, radius in enumerate(radiuses):
#     cur_real_recordings = real_recordings_per_vel[idx_radius]
#     avg_all_channels = np.mean(cur_real_recordings.avg_audio_channels, 0)
#     avg_all_channels_all_radiuses.append(avg_all_channels)
#     plt.plot(cur_real_recordings.encoder_readings_for_avg_audio, avg_all_channels, label=f'Radius {radius}')
# plt.legend(loc='lower right')
# plt.title('Averaged signal over all channels')
# plt.show()
# plt.clf()

# avg_all_channels_all_radiuses = np.array(avg_all_channels_all_radiuses)
# %%
maxes_all = []
for idx_radius, radius in enumerate(radiuses):
    cur_real_recordings = real_recordings_per_vel[idx_radius]
    #maxes = np.take(np.max(cur_real_recordings.avg_audio_channels, 1), [0,1,3])
    maxes = np.max(cur_real_recordings.avg_audio_channels, 1)
    # avg_all_channels = np.mean(cur_real_recordings.avg_audio_channels, 0)
    # avg_all_channels_all_radiuses.append(avg_all_channels)
    maxes_all.append(maxes)

maxes_all = np.array(maxes_all)
radiuses_int = np.array([int(x) for x in radiuses]) - const_diff_mic
radiuses_mat = np.tile(radiuses_int, (maxes_all.shape[1], 1)).T

for i in range(maxes_all.shape[1]):
    plt.scatter(radiuses_mat[:,i], maxes_all[:,i], label=f'Channel {i}', s=8, zorder=2, marker="^")

def func(x, a, b):
    return a / x**b

popt, pcov = curve_fit(func, radiuses_mat.flatten(), maxes_all.flatten(),
                        bounds=([0,1.5], [np.inf,4]))


x_range = np.linspace(min(radiuses_int), max(radiuses_int), 500)
# x_range = np.linspace(min(all_mics_poses_sorted)-5, max(all_mics_poses_sorted)+5, 500)
plt.plot(x_range, func(x_range, *popt), label=f'Fitted to `{round(popt[0],2)} / x**{round(popt[1],2)}`', color='purple', zorder=1, lw=0.5)

plt.title(f'Max Peaks at RPS={round(cur_real_recordings.rps,2)}[Hz], Rotor {rotor_type} {mode}, Fitted to `a / x**b`')
plt.xlabel('Mics Distances [cm]')
plt.legend()
plt.show()
plt.clf()
# %%
