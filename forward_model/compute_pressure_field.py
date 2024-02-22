import os
import numpy as onp
from EARS.localization.phase_modulation.phase_modulator import PhaseModulatorRaw
from EARS.localization.phase_modulation.rir import _fourier_batch_convolve_with_rir, _fourier_batch_convolve_with_rir_padding
import torch
from EARS.localization.physics import PLOT_DT
from tqdm import tqdm
from EARS.pyroomacoustics_differential import consts
import multiprocessing
import matplotlib.pyplot as plt
from math import ceil

'''
def _batch_convolve_with_rir(sound: torch.Tensor, rir: torch.Tensor):
    #Convolve the sound with the rir
    # rir has shape   (num_rirs, num_mics, num_sources, max_rir_length) ~ (10,8,1024,3000)
    # sound has shape (num_sounds, num_sources, max_sound_length) ~ (1,1024, 3000)
    # we want to convolve each sound with all the rirs
    # output should have shape (num_sounds, num_rirs, num_mics, max_rir_length + max_sound_length - 1)

    num_sounds, num_sources, _ = sound.shape
    num_rirs, num_mics, _, max_rir_length = rir.shape

    # More expanded implementation of the last line (advantage: more readable; disadvantage: uses temporary variables which might increase memory consumption):

    # rir = rir.view(num_rirs * num_mics, num_sources, max_rir_length).flip([2])  # shape: (num_rirs * num_mics, num_sources, max_rir_length)
    # #rir = torch.flip(rir, dims=[2])  # shape: (num_rirs * num_mics, num_sources, max_rir_length)
    # output = torch.nn.functional.conv1d(sound, rir, padding=max_rir_length - 1)  # shape: (num_sounds, num_rirs * num_mics, max_sound_length + max_rir_length - 1)
    # output = output.view(num_sounds, num_rirs, num_mics, -1)  # shape: (num_sounds, num_rirs, num_mics, max_sound_length + max_rir_length - 1)
    # return output

    # The following code is equivalent to the above code, but it is faster
    return torch.nn.functional.conv1d(sound,  rir.view(num_rirs * num_mics, num_sources, max_rir_length).flip([2]), padding=max_rir_length - 1).view(num_sounds, num_rirs, num_mics, -1)

'''

def batch_convolve(input_sound, rir):
    num_sounds, num_sources, _ = input_sound.shape
    num_rirs, num_mics, num_sources, max_rir_length = rir.shape
    window_size = ceil(rir.shape[-1]/2)
    input_sound = torch.nn.functional.pad(input_sound, (window_size, window_size), mode='circular').double()
    return torch.nn.functional.conv1d(input_sound, rir.view(num_rirs * num_mics, num_sources, max_rir_length).flip([2]), padding=0).view(num_sounds, num_rirs, num_mics, -1).double()


def get_rir(path):
    rir = onp.load(path)
    rir_torch = torch.from_numpy(rir)
    rir_torch = rir_torch.unsqueeze(1)
    return rir_torch

def convolve(input_sound, rir, batch_size=250):
    final_output = []
    if rir.shape[0] < batch_size:
        batch_size = rir.shape[0]
    for i in range(0, rir.shape[0]//batch_size):
        output = batch_convolve(input_sound=input_sound, rir=rir[i*batch_size:(i+1)*batch_size])
        output = output.squeeze()
        final_output.append(output)
    output = torch.cat(final_output, dim=0)
    #output = _fourier_batch_convolve_with_rir(sound=input_sound, rir=rir)
    #print(f"Output shape: {output.shape}")
    #output = output.squeeze()

    # output = _fourier_batch_convolve_with_rir(sound=input_sound, rir=rir)
    # output = output.squeeze()
    return output

def process_rir(path, input_sound):
    rir_torch = get_rir(path).to(input_sound.device)
    print(f'Computing rir {path}')
    output = convolve(input_sound, rir_torch)
    print(f'Computed rir {path}')
    return output

def build_mask(mics_R=0.9144):
    rotor_length_meter = consts.inch_to_meter(18)
    xs = onp.arange(0, 5.0, 0.02)
    ys = onp.arange(0, 5.0, 0.02)
    mask = onp.ones((len(xs), len(ys)))
    #mics_R = 0.9144
    delta_rad = 1.5 * rotor_length_meter/2 # dist between rotors

    origin2D = onp.array([  [2.5 - delta_rad, 2.5 + delta_rad], [2.5 - delta_rad, 2.5 + delta_rad],
                                [2.5 - delta_rad, 2.5 - delta_rad], [2.5 - delta_rad, 2.5 - delta_rad],
                                [2.5 + delta_rad, 2.5 + delta_rad], [2.5 + delta_rad, 2.5 + delta_rad],
                                [2.5 + delta_rad, 2.5 - delta_rad], [2.5 + delta_rad, 2.5 - delta_rad]])

    print('Creating rotors mask')
    for yi, y in enumerate(ys):
        # y2 = round(y, 2)
        for xi, x in enumerate(xs):
            x2 = round(x, 2)
            if onp.any((x2-origin2D[:,0])**2 + (y-origin2D[:,1])**2 < mics_R**2):
                mask[xi, yi] = 0
    return mask

def translate_recordings_to_dB(recordings_per_channel, cutoff=-4.5):
    # cutoff in dB
    recordings_per_channel_db = onp.maximum(onp.log10(abs(recordings_per_channel).clip(min=10**-9)), cutoff) - cutoff
    recordings_per_channel_db_signed = recordings_per_channel_db * onp.sign(recordings_per_channel)
    vmin = recordings_per_channel_db_signed[~onp.isnan(recordings_per_channel_db_signed)].min()
    vmax = recordings_per_channel_db_signed[~onp.isnan(recordings_per_channel_db_signed)].max()
    return recordings_per_channel_db_signed, vmin, vmax

def process_pressure_field(pressure_field, saving_directory):
    pressure_field_db_signed, vmin, vmax = translate_recordings_to_dB(pressure_field)
    pressure_field_db_signed = pressure_field_db_signed.reshape(250,250,pressure_field_db_signed.shape[-1])

    visualization_directory = os.path.join(saving_directory, 'pf_visualization')
    if not os.path.exists(visualization_directory):
        os.makedirs(visualization_directory)

    for i in tqdm(range(pressure_field_db_signed.shape[-1])):
        plt.imshow(pressure_field_db_signed[...,i], extent=[0.0,5.0,0.0,5.0], vmin=vmin, vmax=vmax)
        plt.savefig(os.path.join(visualization_directory, f'{i}.png'))
        plt.close()
    
    print('Finished computing the visualization')

def process_directory(rir_directory_path, saving_directory_path, input_sound, rotor_position):
    print(f'Reading from: {rir_directory_path}')
    print(f'Saving in: {saving_directory_path}')

    x_axis = onp.arange(0, 5.0, 0.02)
    onp.save(os.path.join(saving_directory_path, 'spatial_locations_x.npy'), x_axis)
    print(f"Spatial locations x saved in {os.path.join(saving_directory_path, 'spatial_locations_x.npy')}")

    y_axis = onp.arange(0, 5.0, 0.02)
    onp.save(os.path.join(saving_directory_path, 'spatial_locations_y.npy'), y_axis)
    print(f"Spatial locations y saved in {os.path.join(saving_directory_path, 'spatial_locations_y.npy')}")

    temporal_positions = onp.arange(0,PLOT_DT*input_sound.shape[-1], PLOT_DT)
    onp.save(os.path.join(saving_directory_path, 'temporal_positions.npy'), temporal_positions)
    print(f"Temporal positions saved in {os.path.join(saving_directory_path, 'temporal_positions.npy')}")

    onp.save(os.path.join(saving_directory_path, 'rotor_position.npy'), rotor_position)
    print(f"Rotor position saved in {os.path.join(saving_directory_path, 'rotor_position.npy')}")

    mask = build_mask(mics_R=0.51) #build_mask(mics_R=0.9144-0.4)
    onp.save(os.path.join(saving_directory_path, 'mask.npy'), mask)
    print(f"Mask saved in {os.path.join(saving_directory_path, 'mask.npy')}")

    # pool = multiprocessing.Pool(processes=int(0.8*os.cpu_count()))
    # grid = pool.starmap(process_rir,
    #                 [(os.path.join(rir_directory_path, rir_path), input_sound) \
    #                  for rir_path in \
    #                     sorted(os.listdir(rir_directory_path), key=lambda x: float(x.strip('.npy')))]
    #                 )
    # pool.close()
    # pool.join()
    # print(f"Grid shape: {len(grid)}")

    print(sorted(os.listdir(rir_directory_path), key=lambda x: float(x.strip('.npy'))))

    '''
    grid = []
    '''
    all_sim_sines = []
    min_len_sim = onp.inf
    for yi, rir_path in tqdm(enumerate(sorted(os.listdir(rir_directory_path), key=lambda x: float(x.strip('.npy'))))):
        #print(f'processing index {yi} rir path {rir_path}')
        rir_torch = get_rir(os.path.join(rir_directory_path, rir_path)).to(input_sound.device)
        output = convolve(input_sound, rir_torch)
        #all_sim_sines.append(output)
        
        cur_simulated_sines = onp.zeros((250,output.shape[-1]))
        cur_simulated_sines[onp.where(mask[:, yi] == 1)] = output.detach().cpu().numpy()
        '''
        try:
            cur_simulated_sines[onp.where(mask[yi, :] == 1)] = output.detach().cpu().numpy()
        except:
            print(f'mask: {onp.where(mask[yi, :] == 1)} (size: {len(onp.where(mask[yi, :] == 1))})')
            print(f'output {output.shape}')
            raise ValueError()
        '''

        all_sim_sines.append(cur_simulated_sines)
        min_len_sim = min(min_len_sim, cur_simulated_sines.shape[-1])
        

        '''
        grid.append(output)
        '''
    
    '''
    shapes = []
    for i,sines in enumerate(all_sim_sines):
        print(f'{i}: {sines.shape}')
        shapes.append(sines.shape)
    shapes = onp.array(shapes)
    onp.save(os.path.join(saving_directory_path, 'our_shapes.npy'), shapes)
    print(f"Our shapes saved in {os.path.join(saving_directory_path, 'our_shapes.npy')}")
    exit()
    '''
    
    simulated_sines_raw = onp.vstack([x[:, :min_len_sim] for x in all_sim_sines])
    simulated_sines,_,_ = translate_recordings_to_dB(simulated_sines_raw)
    grid_numpy = simulated_sines.reshape(250,250,simulated_sines.shape[-1])
    #grid_numpy, vmin, vmax = translate_recordings_to_dB(grid_numpy)
    print(f'final shape: {grid_numpy.shape}')

    '''
    cut = min(rir.shape[-1] for rir in grid)
    grid = [rir[...,:cut] for rir in grid]
    #grid = torch.stack(grid)
    grid_numpy = onp.zeros((250,250,cut))
    #for i, (x,y) in enumerate(sorted(zip(onp.where(mask == 1)[0], onp.where(mask == 1)[0]))):
        #grid_numpy[x,y] = grid[x][y].detach().cpu().numpy()
    '''
        

    '''
    for y in range(250):
        idx = 0
        for x in range(250):
            if mask[x,y] == 1:
                grid_numpy[y,x] = grid[y][idx].detach().cpu().numpy()
                idx += 1
    '''

    #grid_numpy[mask == 1] = *grid
    #grid_numpy = grid.detach().cpu().numpy()
    #assert grid_numpy.shape[0] == 250 and grid_numpy.shape[1] == 250, "Error in dimensions"
    onp.save(os.path.join(saving_directory_path, 'pressure_field.npy'), grid_numpy)
    print(f"Pressure field saved in {os.path.join(saving_directory_path, 'pressure_field.npy')}")

    process_pressure_field(simulated_sines_raw.reshape(250,250,simulated_sines_raw.shape[-1]), saving_directory_path)

    

rir_directory_path_prefix = '/mnt/walkure_public/tomh/rir/entire_room_org_2.5_2.5_rotors_4_final1_fine/'
rir_directory_path_free_space = os.path.join(rir_directory_path_prefix, 'rir_images0')
rir_directory_path_indoor = os.path.join(rir_directory_path_prefix, 'rir_indoor')

saving_directory_path_prefix = '/mnt/walkure_public/tamirs/pressure_field/'
saving_directory_path_free_space = os.path.join(saving_directory_path_prefix, 'free_space_new')
if not os.path.exists(saving_directory_path_free_space):
    os.makedirs(saving_directory_path_free_space)
saving_directory_path_indoor = os.path.join(saving_directory_path_prefix, 'indoor_new')
if not os.path.exists(saving_directory_path_indoor):
    os.makedirs(saving_directory_path_indoor)

phase_modulation_path_prefix = '/mnt/walkure_public/tamirs/phase_modulations/'
# new phase modulations at phase_modulation_separated
# old phase modulations at phases_freeze_iter_5
phase_modulation_path = os.path.join(phase_modulation_path_prefix, 'phase_modulation_separated.npy')
phase_modulation = onp.load(phase_modulation_path)
phase_modulation_torch = torch.from_numpy(phase_modulation)

NUM_ROTORS: int = 4

#rotor_position_path = '/mnt/walkure_public/tamirs/pressure_field/indoor_recordings_4_rotors_4_mics_d_0.05_mode_images0_phases_freeze_iter_5.npy/rotor_position.npy'
#rotor_position_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'encoder_readings.npy')
rotor_position_path = '/mnt/walkure_public/tamirs/encoder_readings.npy'
rotor_position = onp.load(rotor_position_path)
cut = min(phase_modulation.shape[-1], rotor_position.shape[-1])
encoder_readings = rotor_position[:NUM_ROTORS, :cut]
rotor_position = encoder_readings+phase_modulation

# Load the input sound while it's still under the sources format
phase_modulator = PhaseModulatorRaw(modulation=phase_modulation_torch)
input_sound = phase_modulator.get_input_sound()
input_sound = input_sound.unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_sound = input_sound.to(device).detach()
print(input_sound.shape)

#process_directory(rir_directory_path_free_space, saving_directory_path_free_space, input_sound, rotor_position)
process_directory(rir_directory_path_indoor, saving_directory_path_indoor, input_sound, rotor_position)

# iterate over the rirs in the relevant directories:
# for file in directory:
#   - convolve input sound with file
#   - append the output in "grid"
# 
# - build spatial locations
# - build temporal positions (how?)
# - (manually) paste rotor positions over time 