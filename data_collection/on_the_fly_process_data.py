#%% imports

import scipy
import scipy.io.wavfile as wf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pre_processing import *

from create_data_from_recordings import process_enc_z_channels, calculate_rps

#%% params - edit

# files params
data_dir_name = '14_9_120_5' #TODO
rotor_type = 18
mode = 'CCW'
vel_rate = 14 #5

data_file_name = 'rotor_' + str(rotor_type) + '_' + mode

# plotting params
idx_audio_channel_to_plot = 1#TODO
num_audio_channels = 4
partial_plot_time = 0.6
rotor_angle_plot_size_coeff = 0.00001

#%% read metadata mics and recording
df_metadata = pd.read_csv(os.path.join('..', data_dir_name, 'metadata.csv'))
df_metadata_mics = pd.read_csv(os.path.join('..', data_dir_name, 'metadata_mics.csv'))

# sig params
start_time = float(df_metadata[df_metadata['exp_name'] == data_file_name]['start_time'])
time_for_single_vel = 10

# channels params
z_channel_idx = int(df_metadata_mics[df_metadata_mics['function']=='z']['channel']) - 2 # home position
enc_channel_idx = int(df_metadata_mics[df_metadata_mics['function']=='alpha']['channel']) - 2 #1 # encoder by angle

#%% read recorded signal
data_file_path = os.path.join('..', data_dir_name, data_file_name+'.wav')

fs, data = wf.read(data_file_path)
signal_time_sec = data.shape[0] / fs
x_range = np.arange(data.shape[0]) / fs

data_by_channels = data.T[1:,:]

#%% BY ONE VEL

# process data by the velocity time limits
vel_start = int(fs * (start_time + 2 + time_for_single_vel * vel_rate))
vel_end = int(fs * (start_time - 0.5 + time_for_single_vel * (vel_rate+1)))
data_by_channels_by_vel = data_by_channels[:,vel_start:vel_end]
x_range_by_vel = x_range[vel_start:vel_end]

home_pos_channel = data_by_channels_by_vel[z_channel_idx]
enc_channel = data_by_channels_by_vel[enc_channel_idx]

# process rotor angle by the z-channel and the angle channel of the encoder
start_cycle_time, _ = process_enc_z_channels(home_pos_channel, 
                                        x_range, fs, vel_start)
angle_change, sig_sign_enc = process_enc_z_channels(enc_channel, 
                                        x_range, fs, vel_start)

# calculate rounds per second (RPS [Hz])
rps = calculate_rps(home_pos_channel, fs)

print(f'Processing exp with rps={rps}[Hz]')

def convert_mics_pos_to_phase(mics_angular_pos_deg, mode):
    # mics metadata handling
    if mode == 'CCW':
        return np.radians(mics_angular_pos_deg)
    else: # mode == 'CW'
        return np.radians(360 - mics_angular_pos_deg)

data_by_channels_by_vel_2 = np.zeros_like(data_by_channels_by_vel)
for idx_loop in range(num_audio_channels):
    dist_mic = convert_mics_pos_to_phase(df_metadata_mics.loc[idx_loop, 'CW angular position [deg]'], mode)
    data_by_channels_by_vel_2[idx_loop] = np.roll(data_by_channels_by_vel[idx_loop], int((fs * dist_mic) / (np.pi * rps*2)))
data_by_channels_by_vel = data_by_channels_by_vel_2

rotor_angle = np.zeros_like(sig_sign_enc, dtype=np.int)
rotor_angle[(angle_change * fs - vel_start).astype('int')] = 1
home_position = np.zeros_like(sig_sign_enc)
home_position[(start_cycle_time * fs - vel_start).astype('int')] = 1

part_sum = 0
for i in range(len(rotor_angle)):
            
    if home_position[i] == 1:
        part_sum = 0
    elif rotor_angle[i] == 1:
        part_sum += 8

    rotor_angle[i] = part_sum

sig = data_by_channels_by_vel[idx_audio_channel_to_plot]

plot_start = 0
plot_end = int(partial_plot_time * fs)

plt.plot(x_range_by_vel[plot_start:plot_end], sig[plot_start:plot_end], alpha=0.7)

plt.plot(x_range_by_vel[plot_start:plot_end], (rotor_angle * rotor_angle_plot_size_coeff)[plot_start:plot_end], c='orange')
plt.title(f'Audio channel {idx_audio_channel_to_plot-1}, RPS={round(rps,2)}[Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.clf()

#%% FFT BY ONE VEL

# calc fft and show the magnitude spectrum of the signal
fft_sig = scipy.fftpack.rfft(sig)
xf = scipy.fftpack.rfftfreq(len(sig), 1 / fs)

bpf_harmonics = np.arange(rps*2, rps*10, rps*2)

plt.plot(xf, np.abs(fft_sig), zorder=1)
plt.scatter(bpf_harmonics, (np.abs(fft_sig))[8*bpf_harmonics.astype(np.int)], 
            s=80, facecolors='none', edgecolors='r', 
            label='BPF harmonics', zorder=2) #(np.abs(fft_sig))[bpf_harmonics.astype(np.int)], 
plt.scatter(rps, np.abs(fft_sig)[8*int(rps)], 
            s=80, facecolors='none', edgecolors='green', 
            label='RPS', zorder=2)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [-]')
plt.legend()
plt.xlim(left=0, right=300)
plt.title(f'Magnitude Spectrum of audio channel {idx_audio_channel_to_plot-1}, RPS={round(rps,2)}[Hz]')
plt.show()
plt.clf()
#%%
rotor_angle_step = 8

#audio_sample = butter_lowpass_filter(audio_sample, 500, 44100, order=2)

for channel_idx in range(num_audio_channels):

    sig = data_by_channels_by_vel[channel_idx]
    sig = butter_lowpass_filter(sig, 500, fs, 2)
    avg_audio = []
    for i in range(min(rotor_angle), max(rotor_angle), rotor_angle_step):
        avg_audio_ch = np.mean(sig[rotor_angle == i])
        avg_audio.append([i, avg_audio_ch])
        
    avg_audio = np.array(avg_audio)
    rotor_audio = np.sort(np.vstack([rotor_angle, sig]), 0)

    #plt.scatter(rotor_audio[1], rotor_audio[0], s=0.1, zorder=1, label='original signal')
    plt.plot(avg_audio[:,0], avg_audio[:,1], zorder=2, label=f'channel {channel_idx}')
plt.xlabel('Rotor angular position (from 1 to 1024)')
plt.legend()
#plt.ylim(top=0.04, bottom=-0.04)
plt.title(f'Averaged signal, RPS={round(rps,2)}[Hz]')
plt.show()
plt.clf()
#%% make the triangle plot
max_vel_idx = 14
vel_rates = list(range(max_vel_idx))
rps_s = []

for vel_rate in vel_rates:

    vel_start = int(fs * (start_time + 2 + time_for_single_vel * vel_rate))

    vel_end = int(fs * (start_time - 0.5 + time_for_single_vel * (vel_rate+1)))
    data_by_channels_by_vel = data_by_channels[:,vel_start:vel_end]
    x_range_by_vel = x_range[vel_start:vel_end]

    home_pos_channel = data_by_channels_by_vel[z_channel_idx]
    enc_channel = data_by_channels_by_vel[enc_channel_idx]

    # process rotor angle by the z-channel and the angle channel of the encoder
    start_cycle_time, _ = process_enc_z_channels(home_pos_channel, 
                                            x_range, fs, vel_start)
    angle_change, sig_sign_enc = process_enc_z_channels(enc_channel, 
                                            x_range, fs, vel_start)

    # calculate rounds per second (RPS [Hz])
    rps = calculate_rps(home_pos_channel, fs)

    rps_s.append(rps)

plt.scatter(vel_rates, rps_s, c='b', label='rps value')
plt.legend(loc='lower right')
plt.xticks(range(0,max_vel_idx+1,5))
plt.xlabel('Velocity Index')
plt.ylabel('Rounds Per Second [Hz]')
plt.grid()
plt.show()
plt.clf()


# %%
