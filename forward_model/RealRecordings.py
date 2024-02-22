#%%
import pandas as pd
import os, sys
import numpy as np
from itertools import zip_longest
from sklearn.preprocessing import MinMaxScaler
cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from data_processing.pre_processing import *

class RealRecordings:
    def __init__(self, data_dir_path, data_file_name='real_recordings.h5', sample_len=0.5, fs=44100, apply_filter=True, low_cut_coeff=0.8, high_cut_coeff=20.,
                recordings_mode='CCW', rotor_type=15, vel_rate=0, compansate_on_mics_phase=False):
        # sample_len is in seconds

        # recordings metadata
        self.fs = fs
        self.sample_len = sample_len
        self.recording_mode = recordings_mode
        self.rotor_type = rotor_type
        # mics metadata
        self.mics_dist, self.mics_angular_pos, self.mics_num = self.get_mics_metadata(data_dir_path) 
        # recordings metadata and data, #TODO - rps = rotation per sec (angular vel)
        self.rps, self.bpf, self.encoder_readings, self.audio_channels = self.get_recordings_metadata_and_data(data_dir_path, data_file_name, vel_rate)

        # audio processing
        # compensate on mics angular pos (relative phase)
        if compansate_on_mics_phase:
            self.compensate_on_mics_phase()

        if apply_filter:
            self.apply_filter_on_data(low_cut_coeff, high_cut_coeff)
        
        self.create_avg_audio()
    

    def get_recordings_metadata_and_data(self, data_dir_path, data_file_name, vel_rate):
        data_file_path = os.path.join(data_dir_path, data_file_name)
        df_data = pd.read_hdf(data_file_path)
        data = df_data[(df_data['mode']==self.recording_mode) & (df_data['rotor_type']==self.rotor_type) 
                & (df_data['vel_rate']==vel_rate)]
        rps = data['rps'][0]
        bpf = rps * 2 # BPF is defined as `rps * num_blades`. In our case `num_blades == 2`
        
        # get recordings data
        encoder_readings = data['rotor_angle'].to_numpy()[0]
        audio_channels = data['audio_channels'].to_numpy()[0]
        return rps, bpf, encoder_readings, audio_channels


    def compensate_on_mics_phase(self):
        audio_channels_compensated = np.zeros_like(self.audio_channels)
        
        for idx_loop in range(self.audio_channels.shape[0]):
            dist_mic = self.convert_mics_pos_to_phase(self.mics_angular_pos[idx_loop], self.recording_mode)
            audio_channels_compensated[idx_loop] = np.roll(self.audio_channels[idx_loop], int((self.fs * dist_mic) / (np.pi * self.rps*2)))

        self.audio_channels = audio_channels_compensated


    def get_mics_metadata(self, data_dir_path, metadata_mics_file_name='metadata_mics.csv'):
        mics_data_file_path = os.path.join(data_dir_path, metadata_mics_file_name)
        df_mics_data = pd.read_csv(mics_data_file_path)
        mics_channels_metadata = df_mics_data[df_mics_data['function']=='audio']

        mics_dist = mics_channels_metadata['distance from rotor origin [cm]'].to_numpy() * 0.01 # convertion to meters
        mics_angular_pos = np.radians(mics_channels_metadata['CW angular position [deg]'].to_numpy())
        mics_num = mics_channels_metadata['mic number'].to_numpy()
        
        return mics_dist, mics_angular_pos, mics_num


    def convert_mics_pos_to_phase(self, mics_angular_pos_deg, mode):
        # mics metadata handling
        if mode == 'CCW':
            return mics_angular_pos_deg
        else: # mode == 'CW'
            return 2 * np.pi - mics_angular_pos_deg


    def create_avg_audio(self):
        avg_audio_all = []
        rotor_angle_step = 8

        for audio_sample in self.audio_channels:
            avg_audio_single_channel = []

            for i in range(int(min(self.encoder_readings)), int(max(self.encoder_readings)), rotor_angle_step):
                avg_audio_single_channel.append([i, np.mean(audio_sample[self.encoder_readings == i])])

            avg_audio = np.array(avg_audio_single_channel)
            x_range =  2 * np.pi * avg_audio[:,0] / max(self.encoder_readings)
            avg_audio = avg_audio[:,1]
            
            avg_audio_all.append(avg_audio)
        
        self.avg_audio_channels = np.array(avg_audio_all)
        self.encoder_readings_for_avg_audio = x_range


    def get_avg_audio(self, sample_len):
        array_len = self.rps * sample_len
        avg_channels = np.vstack([np.expand_dims(self.encoder_readings_for_avg_audio, axis=0), self.avg_audio_channels])
        # avg_sample = np.tile(avg_channels, int(np.floor(array_len)))
        # avg_sample = np.concatenate((avg_sample, 
        #                         avg_channels[:,:int(avg_channels.shape[1]*(array_len - int(np.floor(array_len))))]), axis=1)
        avg_sample = self.tile_samples(avg_channels, array_len)
        self.avg_audio_fs = self.avg_audio_channels.shape[1] * self.rps # calculate sampling frequency based on RPS and the encoder sampling rate

        return avg_sample, self.avg_audio_fs

    def tile_samples(self, data, array_len):
        avg_sample = np.tile(data, int(np.floor(array_len)))
        avg_sample = np.concatenate((avg_sample, 
                                data[:,:int(data.shape[1]*(array_len - int(np.floor(array_len))))]), axis=1)
        return avg_sample

    def apply_filter_on_data(self, low_cut_coeff, high_cut_coeff, filter_order=1):

        if high_cut_coeff == -1: # apply highpass
            low_cut = low_cut_coeff * self.bpf
            self.audio_channels = butter_lowpass_filter(self.audio_channels, low_cut, self.fs, order=filter_order, by='highpass')
        elif low_cut_coeff == -1: # apply lowpass
            high_cut = self.bpf * high_cut_coeff
            self.audio_channels = butter_lowpass_filter(self.audio_channels, high_cut, self.fs, order=filter_order, by='lowpass')
        else: # apply bandpass filter
            high_cut = self.bpf * high_cut_coeff
            low_cut = low_cut_coeff * self.bpf
            self.audio_channels = butter_bandpass_filter(self.audio_channels, low_cut, high_cut, self.fs, order=filter_order)


class RealRecordingsSeveralExperiments(RealRecordings):
    def __init__(self, data_dir_paths, data_file_name='real_recordings.h5', sample_len=0.5, fs=44100, apply_filter=True, low_cut_coeff=0.8, high_cut_coeff=20.,
                recordings_mode='CCW', rotor_type=18, vel_rate=0, compansate_on_mics_phase=False, num_mics_in_each_exp=4, filter_order=1, 
                channel=-1, num_phases=4, scale_data=False, zigzag_style=False, use_shift_phases=False):
        # sample_len is in seconds
        # channel determines the chosen channel to use to create the pressure field. If none is used (-1) then all recorded channels are being used.

        # recordings metadata
        self.fs = fs
        self.sample_len = sample_len
        self.recording_mode = recordings_mode
        self.rotor_type = rotor_type

        self.use_shift_phases = use_shift_phases

        # process all experiments for metadata and data
        self.process_all_experiments(data_dir_paths, data_file_name, vel_rate, num_mics_in_each_exp, channel, num_phases)
        # audio processing
        # compensate on mics angular pos (relative phase)
        if scale_data:
            scaler = MinMaxScaler(feature_range=(-1,1))
            self.audio_channels = scaler.fit_transform(self.audio_channels)
        
        if compansate_on_mics_phase:
            self.compensate_on_mics_phase()
        if apply_filter:
            self.apply_filter_on_data(low_cut_coeff, high_cut_coeff, filter_order)
        
        if zigzag_style:
            self.use_recordings_in_zigzag_style(num_phases)

        self.create_avg_audio() #TODO - average to avoid noise?

    def use_recordings_in_zigzag_style(self, num_phases):
        num_mics = self.mics_dist.shape[0]
        idxs = np.sort(np.hstack([np.arange(num_mics)[x::num_phases*2] 
                            if x%2==0 
                            else np.arange(num_mics)[x+num_phases::num_phases*2] 
                            for x in np.arange(num_phases)]))
        self.audio_channels = self.audio_channels[idxs,:]
        self.encoder_readings = self.encoder_readings[idxs,:]
        self.mics_angular_pos = self.mics_angular_pos[idxs]
        self.mics_dist = self.mics_dist[idxs]
        self.mics_num = self.mics_num[idxs]

    def process_all_experiments(self, data_dir_paths, data_file_name, vel_rate, num_mics_in_each_exp=4, 
                                    channel=-1, num_phases=4):
        # init
        self.mics_dist, self.mics_angular_pos, self.mics_num, self.rps_all, self.bpf_all, self.encoder_readings, self.audio_channels = [[] for _ in range(7)]
        self.num_mics_in_each_exp = num_mics_in_each_exp
        min_size = np.inf

        for data_dir_path in data_dir_paths:
            mics_dist, mics_angular_pos, mics_num = self.get_mics_metadata(data_dir_path)
            self.mics_dist.extend(mics_dist)
            self.mics_angular_pos.extend(mics_angular_pos)
            self.mics_num.extend(mics_num)

            rps, bpf, encoder_readings, audio_channels = self.get_recordings_metadata_and_data(data_dir_path, data_file_name, vel_rate)
            self.rps_all.append(rps)
            self.bpf_all.append(bpf)
            self.encoder_readings.append(encoder_readings)
            self.audio_channels.append(audio_channels)

            min_size = min(encoder_readings.shape[0], min_size)
        
        # slice channels to create a np for the entire experiments #TODO why?
        for i in range(len(self.encoder_readings)):
            self.encoder_readings[i] = self.encoder_readings[i][:min_size]
            self.audio_channels[i] = self.audio_channels[i][:,:min_size]
        # self.audio_channels shape is (num_mics_in_each_exp*num_experiments, num_samples) (total_channels=num_mics_in_each_exp=4) 
        self.audio_channels = np.vstack(self.audio_channels)
        # self.encoder_readings shape is (num_mics_in_each_exp*num_experiments, num_samples) (num_mics_in_each_exp=4) (num_samples 
        # of encoder readings and audio channels are different!)
        self.encoder_readings = np.repeat(self.encoder_readings, np.ones(len(self.encoder_readings), dtype=int) * num_mics_in_each_exp, axis=0)

        # the 'basic' rps and bpf of the data are the mean values over all the experiments
        self.rps = np.mean(self.rps_all)
        self.bpf = np.mean(self.bpf_all)
        
        # convert mics data to np arrays
        # shape is (num_experiments* num_mics_in_each_exp) (total_channels = num_mics_in_each_exp=4)
        self.mics_dist = np.asarray(self.mics_dist)
        # shape is (num_experiments* num_mics_in_each_exp) (total_channels = num_mics_in_each_exp=4)
        self.mics_angular_pos = np.asarray(self.mics_angular_pos)
        # shape is (num_experiments* num_mics_in_each_exp) (total_channels = num_mics_in_each_exp=4)
        self.mics_num = np.asarray(self.mics_num)
        
       # breakpoint()
        # in case that we want to use a single channel only
        if channel != -1:
            # Once we consider a single channel this is the only true source of new information
            # single_audio_channel_multi_dist has shape (num_experiments, num_samples) 
            single_audio_channel_multi_dist = self.audio_channels[channel::4]
            # Once we consider a single channel this is the only true source of new information
            # encoder_readings has shape (num_experiments, num_samples) (num_samples of encoder_readings and audio_channels_multi_dist are different!)
            encoder_readings = self.encoder_readings[channel::4]
            all_phases = np.linspace(0, 2, num_phases, endpoint=False)
            # mics_radiuses = num_experiments = the number of the different distances where the mics were placed
            mics_radiuses = single_audio_channel_multi_dist.shape[0]
            # by computing self.mics_angular_pos[channel] / np.pi we convert angles from degrees to fractions of pi (e.g. 90 [degrees] = 0.5 [pi])
            # It's different from radians since e.g. 360 [degrees] = 2 [pi] = 2 pi [radians]
            # Probably we use fractions of pi since then we multiply everything by pi at the end (remember that also all_phases is in fractions of pi as 
            # it is defined as np.linspace(0,2) and NOT np.linspace(0,2*np.pi) )
            # Hence we are simulating all of the phases by fixing the position of one mic and rotating the other mics around it.
            # We consider num_phases virtual microphones. 
            # shift_phases will contain the result of this operation but repeated for all the different distances where the mics were placed
            # shift_phases has shape (num_phases * mics_radiuses) (num_phases = number of virtual microphones, mics_radiuses = num_experiments)
            shift_phases = np.tile(self.mics_angular_pos[channel] / np.pi - all_phases, mics_radiuses) #TODO - ask Tom

            if self.recording_mode == 'CW':
                shift_phases *= -1
            # num_phases * mics_radiuses = number of virtual microphones * number of different distances where the mics were placed
            # num_phases corresponds to the number of virtual microphones (it represents the angle of the microphones)
            # mics_radiuses corresponds to the number of different distances where the mics were placed (it represents the distance of the microphones) 
            audio_channels_all_phases_all_dists = np.empty(
                                                (num_phases * mics_radiuses, 
                                                single_audio_channel_multi_dist.shape[1]))

            radius_idx = -1
            for i in range(audio_channels_all_phases_all_dists.shape[0]):
                if i % num_phases == 0:
                    radius_idx += 1

                audio_channels_all_phases_all_dists[i] = np.roll(single_audio_channel_multi_dist[radius_idx], #TODO-ask Tom
                                                                int((self.fs * shift_phases[i]) / self.bpf))
                # encoder_readings_all_phases_all_dists[i] = np.roll(encoder_readings[radius_idx], 
                #                                                 int((self.fs * shift_phases[i]) / self.bpf))

            self.audio_channels = audio_channels_all_phases_all_dists
            #breakpoint()
            if not self.use_shift_phases:
                self.mics_angular_pos = np.tile(all_phases, mics_radiuses) * np.pi
            else:
                self.mics_angular_pos = shift_phases * np.pi
            self.encoder_readings = np.repeat(encoder_readings, num_phases, axis=0)
            self.mics_dist = np.repeat(self.mics_dist[::num_mics_in_each_exp], num_phases)
            self.mics_num = np.tile(np.arange(num_phases), int(self.mics_dist.shape[0]/num_phases))
            
            border = int((self.fs * (num_phases-1)) / self.rps)
            if self.recording_mode == 'CCW':
                self.audio_channels = self.audio_channels[:, border: ]
                self.encoder_readings= self.encoder_readings[:, border: ]
            else:
                self.audio_channels = self.audio_channels[:, :-border ]
                self.encoder_readings= self.encoder_readings[:, :-border ]


    def create_avg_audio(self):
        avg_audio_all = []
        x_range_all = []
        rotor_angle_step = 8
        min_len = np.inf
        # Here we're iterating over both the virtual microphones (hence the virtual phases, i.e. angles) and the true microphones (hence the true distances)
        for audio_sample, encoder_reading in zip(self.audio_channels, self.encoder_readings):
            avg_audio_single_channel = []

            for i in range(int(min(encoder_reading)), int(max(encoder_reading)), rotor_angle_step):
                avg_audio_single_channel.append([i, np.mean(audio_sample[encoder_reading == i])])

            avg_audio = np.array(avg_audio_single_channel)
            x_range =  2 * np.pi * avg_audio[:,0] / max(encoder_reading)
            avg_audio = avg_audio[:,1]
            min_len = min(min_len, len(avg_audio))
            # Given that we're iterating also over the virtual microphones (which simulate the phase)
            # the relationship between two microphones at the same distance but at different phases is that they will just have a shifted version of the same signal
            # and hence a shifted version of the mean of the signal!
            avg_audio_all.append(avg_audio)
            x_range_all.append(x_range)
        # shape of avg_audio_all is (num_phases * mics_radiuses, num_rotor_angle_step_in_one_round)
        self.avg_audio_channels = np.array(list(zip_longest(*avg_audio_all))).T[:,:min_len]
        #breakpoint()
        self.encoder_readings_for_avg_audio = np.array(list(zip_longest(*x_range_all))).T[:,:min_len]


    def get_avg_audio(self, sample_len, channel=-1):
        # array_len counts the number of rounds that we want (it comes from rounds_per_second * sample_time_length_in_seconds)
        # given that avg_audio_channels has shape (num_phases * mics_radiuses, num_rotor_angle_step_in_one_round)
        array_len = self.rps * sample_len
        self.avg_audio_fs = self.avg_audio_channels.shape[1] * self.rps # calculate sampling frequency based on RPS and the encoder sampling rate
        # avg_audio_samples is given by repeating self.avg_audio_channels for array_len times (i.e. for the number of rounds that we want)
        # if array_len is not an integer we also append that fraction of the round of recording to the end of the array 
        avg_audio_samples = self.tile_samples(self.avg_audio_channels, array_len)
        encoder_readings_for_avg_samples = self.tile_samples(self.encoder_readings_for_avg_audio, array_len)
        #breakpoint()
        if channel != -1:
            # According to me this line is not correct: we have already added all of the virtual microphones to the avg_audio_channels
            # hence to pick specific channels you can't skip by 4 but rather you should also skip by the number of virtual microphones (
            # i.e. num_phases) hence [channel::num_phases] but I'm not even sure about this,
            # given that we haven't started rolling the audio samples from 0, hence we would need to take into account also that!

            return avg_audio_samples[channel::4], encoder_readings_for_avg_samples[channel::4], self.avg_audio_fs, \
                    self.mics_dist[channel::4], self.mics_angular_pos[channel::4]

        return avg_audio_samples, encoder_readings_for_avg_samples, self.avg_audio_fs, self.mics_dist, self.mics_angular_pos
    


#%%
if __name__ == "__main__":
    # data_dir_path = os.path.join(cur_dir,'..','..','data','real_recordings','13_7_LD_exp_wav')
    # recordings_dataset = RealRecordings(data_dir_path, rotor_type=16)
    # avg_audio_1sec, avg_audio_fs = recordings_dataset.get_avg_audio(1)

    channel = 0
    radiuses = [str(x) for x in sorted(list(range(70,125,10)) + [75, 85])]
    data_dir_names = ['13_9_'+str(x) for x in radiuses]
    data_dir_paths = [os.path.join(cur_dir,'..','..','data','real_recordings', x) for x in data_dir_names]
    recordings_dataset = RealRecordingsSeveralExperiments(data_dir_paths, compansate_on_mics_phase=False, vel_rate=5, low_cut_coeff=-1, high_cut_coeff=3.25, channel=channel, num_phases=4)
    avg_audio_samples, encoder_readings_for_avg_samples, avg_audio_fs, _, _ = recordings_dataset.get_avg_audio(1)
    print('stop')
       
    for i in range(0,avg_audio_samples.shape[0], 4):
        plt.plot(avg_audio_samples[i][-400:], label=f'{round(recordings_dataset.mics_angular_pos[i],2)}')
    plt.legend()
    plt.show()

    # start_idx = 0
    # #range_channels = range(start_idx,32,4)
    # range_channels = range(start_idx,4,1)
    # for i in range_channels:
    #     plt.plot(avg_audio_samples[i][-400:], label=f'{round(recordings_dataset.mics_angular_pos[i],2)}')
    # plt.legend()
    # plt.title(f'channel {channel}')
    # plt.show()
    
# %%
