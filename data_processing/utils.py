import os, sys, socket
import numpy as np
import pandas as pd

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from data_processing.pre_processing import *


def convert_mics_pos_to_phase(mics_angular_pos_deg, mode):
    # mics metadata handling
    if mode == 'CCW':
        return np.radians(mics_angular_pos_deg)
    else: # mode == 'CW'
        return np.radians(360 - mics_angular_pos_deg)


def read_recordings_from_h5(data_exp_name, fs, rotor_type, mode, vel_rate):
    # read data of recordings from h5 file
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #TODO: change path to a global one
    s.connect(("8.8.8.8", 80))
    server_ip = s.getsockname()[0]
    s.close()

    if server_ip == '132.68.36.17' or server_ip == '132.68.36.147':
        data_dir_path = os.path.join('..','..','data','real_recordings', data_exp_name)
    else:
        raise ValueError('Please define the data path explicitly in utils.py.')

    data_file_path = os.path.join(data_dir_path, 'real_recordings.h5')
    df_data = pd.read_hdf(data_file_path)
    
    # fs = 44100

    # mode = 'CCW' #'CCW' or 'CW' for counter-clockwise or clockwise, respectively.
    # rotor_type = 18
    # vel_rate = 9

    data = df_data[(df_data['mode']==mode) & (df_data['rotor_type']==rotor_type) 
            & (df_data['vel_rate']==vel_rate)]

    rps = data['rps'].to_numpy()[0]
    bpf = 2 * rps
    rotor_angle = data['rotor_angle'].to_numpy()[0].astype(np.int32)[int(fs*1):int(-1*fs)]
    audio_channels = data['audio_channels'].to_numpy()[0][:,int(fs*1):int(-1*fs)]

    mics_data, mics_distances, mics_distances_sorted, mics_distances_sorted_idx = read_mics_metadata(data_dir_path, mode)

    return audio_channels, rotor_angle, mics_data, rps, bpf, mics_distances, mics_distances_sorted, mics_distances_sorted_idx


def read_mics_metadata(data_dir_path, mode):
    metadata_mics_path = os.path.join(data_dir_path, 'metadata_mics.csv')
    df_metadata_mics = pd.read_csv(metadata_mics_path)
    mics_data = df_metadata_mics[df_metadata_mics['function'] == 'audio']
    mics_data.loc[:,'CW angular position [deg]'] = convert_mics_pos_to_phase(mics_data['CW angular position [deg]'], mode)
    mics_distances = mics_data['distance from rotor origin [cm]']
    mics_distances_sorted_idx = np.argsort(mics_distances).to_numpy()
    mics_distances_sorted = np.sort(mics_distances)
    return mics_data, mics_distances, mics_distances_sorted, mics_distances_sorted_idx


#%% compute avg sig and its props
def comp_avg_sig(df_data, chosen_audio_channels, mode, rotor_type, vel_rate, fs):

    data = df_data[(df_data['mode']==mode) & (df_data['rotor_type']==rotor_type) 
            & (df_data['vel_rate']==vel_rate)]
    
    rps = data['rps'].to_numpy()[0]

    rotor_angle = data['rotor_angle'].to_numpy()[0].astype(np.int32)[int(fs*1):int(-1*fs)]
    audio_channels = data['audio_channels'].to_numpy()[0][:,int(fs*1):int(-1*fs)]

    # create avg audio for the four mics

    avg_audio_all = []
    rotor_angle_step = 8

    for channel_id_loop in chosen_audio_channels:
        avg_audio_single_channel = []
        avg_audio_sample = audio_channels[channel_id_loop]

        # apply lowpass
        avg_audio_sample = butter_lowpass_filter(avg_audio_sample, 500, 44100, order=2)
        avg_audio_sample = butter_lowpass_filter(avg_audio_sample, rps, 44100, order=2, by='highpass')

        for i in range(min(rotor_angle), max(rotor_angle), rotor_angle_step):
            avg_audio_single_channel.append([i, np.mean(avg_audio_sample[rotor_angle == i])])

        avg_audio = np.array(avg_audio_single_channel)
        x_range =  2 * np.pi * avg_audio[:,0] / max(rotor_angle)
        avg_audio = avg_audio[:,1]

        avg_audio_all.append(avg_audio)


    # peaks vs radial coord
    avg_audio_all_np = np.array(avg_audio_all)

    maxes = np.max(avg_audio_all_np, 1)
    mins = np.min(avg_audio_all_np, 1)

    integrals = np.trapz(avg_audio_all_np ** 2, x_range)

    return avg_audio_all_np, maxes, mins, integrals, rps