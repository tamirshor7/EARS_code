import scipy
import scipy.io.wavfile as wf
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
# import librosa
# from librosa import display
from scipy import fftpack
# from scipy.signal import argrelextrema, butter, lfilter, freqz, filtfilt
from pre_processing import *



def get_voltage_from_pwm(peak_motor_voltage=14., signal_period=20.):
    # The pwm is a periodic signal with the period of 20 miliseconds where it's HIGH beween 1-2 milisec.
    # The first given value is 1.3 milisec, next with delta of 0.06 till 1.84 and backwards.
    # Motor peak voltage: 14 Volt
    duty_cycle = np.arange(1.3, 1.84, step=0.06) / signal_period
    voltage = peak_motor_voltage * duty_cycle
    return np.concatenate([voltage, np.flip(voltage)])


def read_wav_file(data_path):
    fs, data = wf.read(data_path)
    data_by_channels = data.T
    x_range = np.arange(data.shape[0]) / fs

    return x_range, data_by_channels, fs

def calculate_rps(enc_channel, fs, bpf = False):
    fft_enc = scipy.fftpack.rfft(enc_channel)
    xf = scipy.fftpack.rfftfreq(len(enc_channel), 1 / fs)
    if bpf:
        #return xf[np.argmax(np.abs(fft_enc))] / 2
        plt.plot(xf, np.abs(fft_enc))
        plt.xlim(left=0, right=200)
        plt.show()
    return xf[np.argmax(np.abs(fft_enc))] * 2 # we multiply the max value by two, because of our signal's characteristic

def process_enc_z_channels(sig, x_range, fs, vel_start):
    # calculate the sign of the signal, and produce the start cycle timings:
    # the home position (z-channel) or the change in angle (enc-channel).
    sig = sig/np.linalg.norm(sig)
    sig_sign = np.clip(np.sign(sig), 0, np.max(sig))
    grad = np.gradient(sig_sign * (1/np.max(sig)))
    start_cycle_time = vel_start/fs + x_range[np.where(grad != 0)[0]][::2]
    return start_cycle_time, sig_sign


def create_dataset(ddir, save_path, num_vels, time_for_single_vel, z_channel, enc_channel, audio_channel_start, audio_channel_end, rotor_types=['18']):
    # read metadata from csv
    print(f'Reading metadata')
    df_metadata = pd.read_csv(os.path.join(ddir, 'metadata.csv'))

    # create an empty df to save the processed data
    df_data = pd.DataFrame(columns = ('mode', 'rotor_type', 'vel_rate', 'voltage_rates', 'rps', 'vel_time_limits',
                                         'rotor_angle', 'audio_channels'))
    print(f'Finished reading metadata')
    # define const data parameters
    # z_channel = 6
    # enc_channel = 7
    # time_for_single_vel = 10

    # modes = ['L', 'C', 'CO'] # circular (C) or cirtuclar_opposite (CO) or linear (L) mode
    modes = ['CW', 'CCW']
    vel_rates = list(range(num_vels))
    voltage_rates = get_voltage_from_pwm()
    # mode = 'L' 
    # rotor_type = '00'
    # vel_rate = 7

    for mode in modes:
        for rotor_type in rotor_types:
            
            # if mode == 'CCW' and rotor_type == '00':
            #     continue

            # read data and meta data
            #exp_name = f'8_3_{rotor_type}_{mode}'
            print(f'Reading data mode {mode}, rotor type {rotor_type}')
            exp_name = f'rotor_{rotor_type}_{mode}'
            data_path = os.path.join(ddir, f'{exp_name}.wav')
            x_range, data_by_channels, fs = read_wav_file(data_path)

            # audio_sample = data_by_channels[3]
            # spectro = convert_to_spectrogram(audio_sample/np.linalg.norm(audio_sample), sr=fs)
            # plot_spectrogram(spectro, fs)
            # plt.show()

            for vel_rate, voltage_rate in zip(vel_rates, voltage_rates):
                print(f'vel_rate: {vel_rate}')
                start_time = df_metadata.loc[df_metadata['exp_name'] == exp_name]['start_time'].to_numpy()[0]

                # process data by the velocity time limits
                vel_start = int(fs * (start_time + 2 + time_for_single_vel * vel_rate))
                vel_end = int(fs * (start_time - 0.5 + time_for_single_vel * (vel_rate+1)))
                data_by_channels_vel = data_by_channels[:,vel_start:vel_end]

                # process rotor angle by the z-channel and the angle channel of the encoder
                start_cycle_time, _ = process_enc_z_channels(data_by_channels_vel[z_channel], 
                                                        x_range, fs, vel_start)
                angle_change, sig_sign_enc = process_enc_z_channels(data_by_channels_vel[enc_channel], 
                                                        x_range, fs, vel_start)

                # calculate rounds per second (RPS [Hz])
                rps = calculate_rps(data_by_channels_vel[z_channel], fs)
                #calculate_rps(data_by_channels_vel[2], fs, True)

                print(f'Processing exp {exp_name} with rps={rps}[Hz]')
                # rps2 = len(start_cycle_time) / 4
                # print(f'rps2={rps2}')
                rotor_angle = np.zeros_like(sig_sign_enc)
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

                #print(f'max rotor angle: {max(rotor_angle)}')

                # shift data by time to start in home position
                shift_start_time = np.where(home_position == 1)[0][0]
                rotor_angle = rotor_angle[shift_start_time:]
                data_by_channels_vel = data_by_channels_vel[:, shift_start_time:]

                # plt.plot(rotor_angle)
                # plt.show()
                
                # add new processed data to the data df
                df_cur = pd.DataFrame({'mode':mode, 'rotor_type':int(rotor_type), 'vel_rate':vel_rate, 'voltage_rates':voltage_rate, 
                                        'rps':rps, 'vel_time_limits':[np.array([vel_start+shift_start_time, vel_end])/fs],
                                        'rotor_angle':[rotor_angle], 
                                        'audio_channels': [data_by_channels_vel[audio_channel_start:audio_channel_end]]})
                                        
                df_data = pd.concat([df_data, df_cur])
    
    # save data to h5
    df_data.to_hdf(save_path, key='df', mode='w')
    # df_data_read = pd.read_hdf(save_path)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-exp_name', default='13_9_80', type=str)
    # parser.add_argument('-num_vels', default=15, type=int)
    # parser.add_argument('-time_for_single_vel', default=10, type=int)
    # parser.add_argument('-z_channel', default=6, type=int)
    # parser.add_argument('-enc_channel', default=7, type=int)
    # parser.add_argument('-audio_channel_start', default=1, type=int)
    # parser.add_argument('-audio_channel_end', default=5, type=int)
    # args = parser.parse_args()

    # ddir = os.path.join('..', args.exp_name)
    # save_path = os.path.join('..', args.exp_name, 'real_recordings.h5')
    # create_dataset(ddir, save_path, args.num_vels, 
    #                 args.time_for_single_vel, args.z_channel, args.enc_channel, args.audio_channel_start, args.audio_channel_end)

    data_dir_name_prefix = '13_9'
    radiuses = [str(x) for x in sorted(list(range(70,125,10)) + [75, 85])]
    
    time_for_single_vel = 10
    audio_channel_start = 1
    audio_channel_end = 5
    z_channel = 6
    enc_channel = 7
    num_vels = 15

    for radius in radiuses:
        
        print('______________________')
        print(f'Extracting recordings of distance {radius}[cm]')

        exp_name = data_dir_name_prefix + '_' + radius
        ddir = os.path.join('..', exp_name)
        save_path = os.path.join('..', exp_name, 'real_recordings.h5')
        create_dataset(ddir, save_path, num_vels, 
                time_for_single_vel, z_channel, enc_channel, audio_channel_start, audio_channel_end)
