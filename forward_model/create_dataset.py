import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from RealRecordings import RealRecordingsSeveralExperiments
from itertools import product
from torch import from_numpy, save
from numpy import savez_compressed, float32
from tqdm import tqdm

channel_values = list(range(4))
recording_mode_values = ['CW', 'CCW']
vel_rate_values = list(range(15))

radiuses = [str(x) for x in sorted(list(range(70,125,10)) + [75, 85])]
data_dir_names = ['13_9_'+str(x) for x in radiuses]
data_dir_paths = [os.path.join(cur_dir,'..','..','data','real_recordings', x) for x in data_dir_names]
output_dir_path = os.path.join(cur_dir,'..','..','data','dataset/')
os.makedirs(output_dir_path, exist_ok=True)

# This is a constant that it is used to compute max_rec_len (it has also been saved in const_params_compressed.npz)
# It only depends on the rir
max_len_rir = 95
# Duration of the time signal
signals_duration = 0.5

for i,(channel, recording_mode, vel_rate) in tqdm(enumerate(product(channel_values, recording_mode_values, vel_rate_values))):
    recordings_dataset = RealRecordingsSeveralExperiments(channel=channel, recordings_mode=recording_mode, vel_rate=vel_rate, 
    data_dir_paths=data_dir_paths, low_cut_coeff=-1, high_cut_coeff=3.25, rotor_type=18, filter_order=1, 
    num_phases = 64)
    real_recordings, encoder_readings, fs, _, _ = recordings_dataset.get_avg_audio(signals_duration)
    omega = recordings_dataset.bpf

    # Things that need to be stored in each datasample
    # max_rec_len, omega, real_recordings, fs, encoder_readings

    max_signal_len = int(fs * signals_duration)
    max_rec_len = max_len_rir + max_signal_len
    if max_rec_len % 2 == 1:
        max_rec_len += 1
    
    '''
    encoder_readings will be given as input only to the neural network, whereas
    the other parameters will be given as input only to the forward function
    (the forward function requires other parameters as well, but they are constant and they
    are saved in const_params_compressed.npz)
    Hence we will convert encoder_readings to a Tensor (currently it's a numpy array) 
    and we will save the rest as it is
    '''

    # Convert encoder_readings to Tensor
    encoder_readings = from_numpy(encoder_readings.astype(float32))

    base_filename = f'{i}chan{channel}_recmode{recording_mode}_vel{vel_rate}'
    save(encoder_readings, output_dir_path + base_filename+'_encoder.pt')
    savez_compressed(output_dir_path + base_filename+'.npz', max_rec_len = max_rec_len, 
                    omega=omega, real_recordings=real_recordings.astype(float32), fs=fs)

