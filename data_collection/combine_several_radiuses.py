#%% imports

import scipy
import scipy.io.wavfile as wf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pre_processing import *

from create_data_from_recordings import process_enc_z_channels, calculate_rps

#%% params
data_dir_name_prefix = '13_9'
rotor_type = 18
mode = 'CCW'

radiuses = [str(x) for x in sorted(list(range(70,125,10)) + [75, 85])]
vel_rates = np.arange(15)

rotor_angle_step = 8
num_audio_channels = 4
time_for_single_vel = 10
data_file_name = 'rotor_' + str(rotor_type) + '_' + mode
radiuses_real_value = [int(x)-17 for x in radiuses]

#%% metadata about channels, assuming no change in the channels between experiments
data_dir_name = data_dir_name_prefix + '_' + str(radiuses[0])
df_metadata_mics = pd.read_csv(os.path.join('..', data_dir_name, 'metadata_mics.csv'))
z_channel_idx = int(df_metadata_mics[df_metadata_mics['function']=='z']['channel']) - 2 # home position
enc_channel_idx = int(df_metadata_mics[df_metadata_mics['function']=='alpha']['channel']) - 2 #1 # encoder by angle

#%% differences between angular velocities
angular_velocities = np.zeros((len(vel_rates), len(radiuses)))

for radius_i, radius in enumerate(radiuses):
    data_dir_name = data_dir_name_prefix + '_' + radius
    data_file_path = os.path.join('..', data_dir_name, data_file_name+'.wav')

    df_metadata = pd.read_csv(os.path.join('..', data_dir_name, 'metadata.csv'))
    start_time = float(df_metadata[df_metadata['exp_name'] == data_file_name]['start_time'])

    # read recorded signal
    fs, data = wf.read(data_file_path)
    signal_time_sec = data.shape[0] / fs
    x_range = np.arange(data.shape[0]) / fs

    data_by_channels = data.T[1:,:]

    for vel_rate in vel_rates:
        # process data by the velocity time limits
        vel_start = int(fs * (start_time + 2 + time_for_single_vel * vel_rate))
        vel_end = int(fs * (start_time - 0.5 + time_for_single_vel * (vel_rate+1)))
        data_by_channels_by_vel = data_by_channels[:,vel_start:vel_end]
        x_range_by_vel = x_range[vel_start:vel_end]

        home_pos_channel = data_by_channels_by_vel[z_channel_idx]

        # calculate rounds per second (RPS [Hz])
        rps = calculate_rps(home_pos_channel, fs)
        angular_velocities[vel_rate, radius_i] = rps

for vel_rate in vel_rates:
    plt.plot(radiuses_real_value, angular_velocities[vel_rate,:], label=f'{vel_rate}')

plt.xlabel('Mics Distance [cm]')
plt.ylabel('Rotations Per Second (RPS)')
plt.title('Angular Velocities at All Experiments')
plt.grid()
plt.legend(loc='right')
plt.show()
plt.clf()

plt.scatter(np.mean(angular_velocities, axis=1), np.std(angular_velocities, axis=1))
plt.title('STD of the angular velocities between the experiments')
plt.ylabel('STD')
plt.xlabel('Mean Rotations Per Second (RPS)')
plt.grid()
plt.show()
plt.clf()

# %%
