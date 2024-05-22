import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from EARS.localization.physics import PLOT_DT
import argparse

def inch_to_meter(inch_val):
    return inch_val * 0.0254

def build_mask(num_rotors, mics_R=0.9144, org_x = 2.5, org_y = 2.5):
    rotor_length_meter = inch_to_meter(18)
    xs = np.arange(0, 5.0, 0.02)
    ys = np.arange(0, 5.0, 0.02)
    mask = np.ones((len(xs), len(ys)))
    #mics_R = 0.9144
    delta_rad = 1.5 * rotor_length_meter/2 # dist between rotors

    if num_rotors == 1: # single rotor
        origin2D = np.array([[org_x, org_y], [org_x, org_y]])
    elif num_rotors == 2: # two rotors
        delta_rad = 1.5 * rotor_length_meter/2 # dist between rotors
        #delta_rad = args['second_rad'] + 0.125
        origin2D = np.array(   [[org_x - delta_rad, org_y], [org_x - delta_rad, org_y],
                                [org_x + delta_rad, org_y], [org_x + delta_rad, org_y]])
    elif num_rotors == 4: # four rotors
        delta_rad = 1.5 * rotor_length_meter/2 # dist between rotors
        origin2D = np.array([  [org_x - delta_rad, org_y + delta_rad], [org_x - delta_rad, org_y + delta_rad],
                                [org_x - delta_rad, org_y - delta_rad], [org_x - delta_rad, org_y - delta_rad],
                                [org_x + delta_rad, org_y + delta_rad], [org_x + delta_rad, org_y + delta_rad],
                                [org_x + delta_rad, org_y - delta_rad], [org_x + delta_rad, org_y - delta_rad]])
    # rotors order by position:
        # 0 2
        # 1 3
    
    else: # not supported
        raise ValueError(f'{num_rotors} rotors are not supported. Use only 1,2,4 rotors.')

    print('Creating rotors mask')
    for yi, y in enumerate(ys):
        # y2 = round(y, 2)
        for xi, x in enumerate(xs):
            x2 = round(x, 2)
            if np.any((x2-origin2D[:,0])**2 + (y-origin2D[:,1])**2 < mics_R**2):
                mask[xi, yi] = 0
    return mask

def translate_recordings_to_dB(recordings_per_channel, cutoff=-4.5):
    # cutoff in dB
    recordings_per_channel_db = np.maximum(np.log10(abs(recordings_per_channel).clip(min=10**-9)), cutoff) - cutoff
    recordings_per_channel_db_signed = recordings_per_channel_db * np.sign(recordings_per_channel)
    vmin = recordings_per_channel_db_signed[~np.isnan(recordings_per_channel_db_signed)].min()
    vmax = recordings_per_channel_db_signed[~np.isnan(recordings_per_channel_db_signed)].max()
    return recordings_per_channel_db_signed, vmin, vmax

def build_rotor_position(num_rotors, number_of_samples, phase_modulation=None):
    if phase_modulation is not None:
        raise NotImplementedError("We still need to deal with the case that uses a phase modulation")
    assert num_rotors in [1,4], f"The current supported number of rotors are 1,4 but you chose {num_rotors}, please set it to a valid value"

    encoder_readings_path = '/mnt/walkure_public/tamirs/encoder_readings.npy'
    encoder_readings = np.load(encoder_readings_path)

    if phase_modulation is not None:
        cut = min(number_of_samples, phase_modulation.shape[-1])
        raise NotImplementedError("We still need to deal with the case that uses a phase modulation")
    else:
        cut = number_of_samples
    
    encoder_readings = encoder_readings[:num_rotors, :cut]

    if phase_modulation is not None:
        rotor_position = encoder_readings + phase_modulation
        raise NotImplementedError("We still need to deal with the case that uses a phase modulation")
    else:
        rotor_position = encoder_readings
    
    return rotor_position

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['free_space', 'indoor'], help='Choose between free_space and indoor')
    parser.add_argument('--path', type=str, help='Path to the data to load')

    args = parser.parse_args()
    return vars(args)

def get_number_rotors(root_path, mode_path):
    paths = os.listdir(root_path)
    i = 0
    while not (paths[i].endswith('.npy') and paths[i].startswith(mode_path)):
        i+= 1
    number_rotors = np.load(os.path.join(root_path, paths[i])).shape[0]
    assert number_rotors in [1,4], f"Error: got {number_rotors} rotors in {os.path.join(root_path, paths[i])}, but supported only 1,4"
    return number_rotors

if __name__ == '__main__':
    args = parse_arguments()

    root_path:str = os.path.join("/mnt", "walkure_public", "tamirs", "simulator", "raw_data", args['path'])
    #root_path: str = "/mnt/walkure_public/tamirs/simulator/raw_data/entire_room_org_2.5_2.5_rotors_4_free_space_and_indoor/"
    assert os.path.exists(root_path), f"Given the loading path {root_path}, but it doesn't exist! Please set a different --path!"

    saving_root_path: str = os.path.join('/mnt', 'walkure_public', 'tamirs', 'simulator', 'processed_data')
    
    EXPERIMENT_NAME: str = args['path'] #'entire_room_org_2.5_2.5_rotors_4_free_space_and_indoor'
    
    saving_root_path = os.path.join(saving_root_path, EXPERIMENT_NAME, f"{args['mode']}__")
    if not os.path.exists(saving_root_path):
        os.makedirs(saving_root_path)

    x_axis = np.arange(0, 5.0, 0.02)
    np.save(os.path.join(saving_root_path, 'spatial_locations_x.npy'), x_axis)
    print(f"Spatial locations x saved in {os.path.join(saving_root_path, 'spatial_locations_x.npy')}")

    y_axis = np.arange(0, 5.0, 0.02)
    np.save(os.path.join(saving_root_path, 'spatial_locations_y.npy'), y_axis)
    print(f"Spatial locations y saved in {os.path.join(saving_root_path, 'spatial_locations_y.npy')}")

    if args['mode'] == 'free_space':
        MODE = 'images0_y_'
    elif args['mode'] == 'indoor':
        MODE = 'indoor_y_'
    else:
        raise ValueError('You need to set --mode to either free_space or indoor')
    
    NUM_ROTORS:int = get_number_rotors(root_path=root_path, mode_path=MODE)
    START_REVOLUTION:int = 2
    WINDOW_LEN:int = 128
    NUM_REVOLUTIONS:int = 8
    
    first_sample:int = WINDOW_LEN*START_REVOLUTION
    last_sample:int = WINDOW_LEN * (START_REVOLUTION+NUM_REVOLUTIONS)
    number_of_samples:int = last_sample-first_sample

    temporal_positions = np.arange(0,PLOT_DT*number_of_samples, PLOT_DT)
    np.save(os.path.join(saving_root_path, 'temporal_positions.npy'), temporal_positions)
    print(f"Temporal positions saved in {os.path.join(saving_root_path, 'temporal_positions.npy')}")

    rotor_position = build_rotor_position(num_rotors=NUM_ROTORS, number_of_samples=number_of_samples, phase_modulation=None)
    np.save(os.path.join(saving_root_path, 'rotor_position.npy'), rotor_position)
    print(f"Rotor position saved in {os.path.join(saving_root_path, 'rotor_position.npy')}")

    mask = build_mask(num_rotors=NUM_ROTORS, mics_R=0.51)
    grid = []
    cut = np.inf
    directory = filter(lambda x: x.endswith('.npy') and x.startswith(MODE), os.listdir(root_path))
    directory = sorted(directory, key= lambda x: float(x.lstrip(MODE).rstrip('.npy')))
    print(f"Iterating over {root_path} with mode {MODE} (length: {len(directory)})")
    print(directory)

    for yi, path in tqdm(enumerate(directory)):
        
        pressure_field = np.load(os.path.join(root_path, path))
        assert pressure_field.shape[0] == NUM_ROTORS, f"Attention the array has a shape {pressure_field.shape} while we expected it to have at the first axis a dimension of {NUM_ROTORS}"
        # sum over rotor dimension
        #window_len*start_revolution:window_len * (start_revolution+num_revolutions)
        pressure_field = pressure_field[..., first_sample:last_sample]
        pressure_field = np.sum(pressure_field, axis=0)
        cur_simulated_sines = np.zeros((250,pressure_field.shape[-1]))
        cur_simulated_sines[np.where(mask[:, yi] == 1)] = pressure_field
        grid.append(cur_simulated_sines)
        cut = min(cut, cur_simulated_sines.shape[-1])

    grid = [pressure_field[..., :cut] for pressure_field in grid]
    grid = np.stack(grid, axis=0)
    print(f"Final grid shape {grid.shape}")

    grid_saving_path: str = os.path.join(saving_root_path, f'pressure_field.npy')
    np.save(grid_saving_path, grid)
    print(f'Grid saved in {grid_saving_path}')

    grid_db, vmin, vmax = translate_recordings_to_dB(grid)
    grid_db_saving_path: str = os.path.join(saving_root_path, f'pressure_field_db.npy')
    np.save(grid_db_saving_path, grid_db)
    print(f'Grid db saved in {grid_db_saving_path}')

    # save a frame for sanity check
    print('Creating frames...')
    image_saving_path:str = os.path.join(saving_root_path, f'frames')

    if not os.path.exists(image_saving_path):
        os.makedirs(image_saving_path)

    for i in tqdm(range(grid_db.shape[-1])):
        plt.imshow(grid_db[..., i], extent=[0.0,5.0,0.0,5.0], vmin=vmin, vmax=vmax, origin='lower')
        plt.colorbar()
        plt.clim(vmin, vmax)
        plt.savefig(os.path.join(image_saving_path, f"frame_{i}.png"), dpi=200)
        plt.clf()
        plt.close()

    print(f'Frames saved in {image_saving_path}')