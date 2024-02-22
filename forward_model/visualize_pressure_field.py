import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from EARS.pyroomacoustics_differential import consts

def translate_recordings_to_dB(recordings_per_channel, cutoff=-4.5):
    # cutoff in dB
    recordings_per_channel_db = np.maximum(np.log10(abs(recordings_per_channel).clip(min=10**-9)), cutoff) - cutoff
    recordings_per_channel_db_signed = recordings_per_channel_db * np.sign(recordings_per_channel)
    vmin = recordings_per_channel_db_signed[~np.isnan(recordings_per_channel_db_signed)].min()
    vmax = recordings_per_channel_db_signed[~np.isnan(recordings_per_channel_db_signed)].max()
    return recordings_per_channel_db_signed, vmin, vmax    

'''
def process_pressure_field(pressure_field, saving_directory, rotor_position, use_log_mapping=False):
    if use_log_mapping:
        pressure_field_db_signed, vmin, vmax = translate_recordings_to_dB(pressure_field)
    else:
        pressure_field_db_signed = pressure_field
        vmin, vmax = pressure_field.min(), pressure_field.max()
    pressure_field_db_signed = pressure_field_db_signed.reshape(250,250,pressure_field_db_signed.shape[-1])

    visualization_directory = os.path.join(saving_directory, 'pf_visualization')
    if not os.path.exists(visualization_directory):
        os.makedirs(visualization_directory)

    for i in tqdm(range(479, pressure_field_db_signed.shape[-1])):
        plt.imshow(pressure_field_db_signed[...,i], extent=[0.0,5.0,0.0,5.0], vmin=vmin, vmax=vmax)
        rotors = get_rotors(rotor_position[...,i])
        for rotor in rotors:
            plt.gca().add_patch(rotor)
        plt.savefig(os.path.join(visualization_directory, f'{i}.png'))
        plt.close()
'''

def process_pressure_field(pressure_field, saving_directory,phases):
    #pressure_field_db_signed, vmin, vmax = translate_recordings_to_dB(pressure_field)
    pressure_field_db_signed = pressure_field.reshape(250,250,pressure_field.shape[-1])

    rotor_centers = ([150,150],[150,100],[100,100],[100,150])
    visualization_directory = os.path.join(saving_directory, 'pf_visualization')
    if not os.path.exists(visualization_directory):
        os.makedirs(visualization_directory)

    for i in tqdm(range(pressure_field_db_signed.shape[-1])):

        line_length  =3
        for rotor,center in enumerate(rotor_centers):
            angle_rads = phases[rotor,i]
            x,y = center
            x1 = int(x - (line_length * np.cos(angle_rads)))
            y1 = int(y - (line_length * np.sin(angle_rads)))
            x2 = int(x + (line_length * np.cos(angle_rads)))
            y2 = int(y + (line_length * np.sin(angle_rads)))
            plt.plot([x1,x2], [y1,y2], color="white", linewidth=3)


        plt.imshow(pressure_field_db_signed[...,i], origin='lower')
        plt.savefig(os.path.join(visualization_directory, f'{i}.png'))
        plt.close()

if __name__ == '__main__':

    NUM_ROTORS:int = 4

    phase_modulation_path_prefix = '/mnt/walkure_public/tamirs/phase_modulations/'
    # new phase modulations at phase_modulation_separated
    # old phase modulations at phases_freeze_iter_5
    phase_modulation_path = os.path.join(phase_modulation_path_prefix, 'phase_modulation_separated.npy')
    phase_modulation = np.load(phase_modulation_path)
    phase_modulation_torch = torch.from_numpy(phase_modulation)

    rotor_position_path = '/mnt/walkure_public/tamirs/encoder_readings.npy'
    rotor_position = np.load(rotor_position_path)
    cut = min(phase_modulation.shape[-1], rotor_position.shape[-1])
    encoder_readings = rotor_position[:NUM_ROTORS, :cut]
    rotor_position = encoder_readings+phase_modulation

    saving_directory_path_prefix = '/mnt/walkure_public/tamirs/pressure_field/'

    saving_directory_path_free_space = os.path.join(saving_directory_path_prefix, 'free_space_new')
    pressure_field_free_space = np.load(os.path.join(saving_directory_path_free_space, 'pressure_field.npy'))
    process_pressure_field(pressure_field_free_space, saving_directory_path_free_space, rotor_position)
    del pressure_field_free_space

    saving_directory_path_indoor = os.path.join(saving_directory_path_prefix, 'indoor_new')
    pressure_field_indoor = np.load(os.path.join(saving_directory_path_indoor, 'pressure_field.npy'))
    process_pressure_field(pressure_field_indoor, saving_directory_path_indoor, rotor_position)
    del pressure_field_indoor


