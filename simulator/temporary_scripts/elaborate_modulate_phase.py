import os
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from EARS.localization.physics import PLOT_DT
import argparse
import torch
from EARS.localization.phase_modulation.phase_modulation_injector import _apply_modulation
import cv2

from EARS.io import hdf5



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

    print(f"Rotor positioned at {origin2D}")

    print('Creating rotors mask')
    for yi, y in enumerate(ys):
        # y2 = round(y, 2)
        for xi, x in enumerate(xs):
            x2 = round(x, 2)
            if np.any((x2-origin2D[:,0])**2 + (y-origin2D[:,1])**2 < mics_R**2):
                mask[xi, yi] = 0
    return mask, origin2D

def translate_recordings_to_dB(recordings_per_channel, cutoff=-4.5):
    # cutoff in dB
    recordings_per_channel_db = np.maximum(np.log10(abs(recordings_per_channel).clip(min=10**-9)), cutoff) - cutoff
    recordings_per_channel_db_signed = recordings_per_channel_db * np.sign(recordings_per_channel)
    vmin = recordings_per_channel_db_signed[~np.isnan(recordings_per_channel_db_signed)].min()
    vmax = recordings_per_channel_db_signed[~np.isnan(recordings_per_channel_db_signed)].max()
    return recordings_per_channel_db_signed, vmin, vmax

def build_rotor_position(num_rotors, number_of_samples, phase_modulation=None):
    assert num_rotors in [1,4], f"The current supported number of rotors are 1,4 but you chose {num_rotors}, please set it to a valid value"

    encoder_readings_path = '/mnt/walkure_public/tamirs/encoder_readings.npy'
    encoder_readings = np.load(encoder_readings_path)

    if phase_modulation is not None:
        assert phase_modulation.shape[-1] == number_of_samples, f"The time dimension of the phase modulation must match the number of samples, but got phase_modulation: {phase_modulation.shape} and number of samples: {number_of_samples}"
        cut = number_of_samples
        #cut = min(number_of_samples, phase_modulation.shape[-1])
    else:
        cut = number_of_samples
    
    encoder_readings = encoder_readings[:num_rotors, :cut]
    cut_phase_modulation = phase_modulation[..., :cut]

    if num_rotors == 4:
        # flip so that it is CW CCW CCW CW --> 1 0 0 1
        encoder_readings[0,:] = encoder_readings[0, ::-1]
        encoder_readings[3,:] = encoder_readings[3, ::-1]

    if phase_modulation is not None:
        rotor_position = encoder_readings + cut_phase_modulation
    else:
        rotor_position = encoder_readings
    
    return rotor_position

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['free_space', 'indoor'], help='Choose between free_space and indoor')
    parser.add_argument('--path', type=str, help='Path to the data to load')
    parser.add_argument('--phase_modulation', type=str, help="Choose the phase modulation between constant_offset, time_varying_sine or provide the path to a phase modulation")
    parser.add_argument('--org_x', type=float, default=2.5, help='x-coordinate where the center of the drone has been put')
    parser.add_argument('--org_y', type=float, default=2.5, help='x-coordinate where the center of the drone has been put')


    args = parser.parse_args()
    return vars(args)

def get_number_rotors(root_path, mode_path):
    paths = os.listdir(root_path)
    i = 0
    while not (paths[i].endswith('.hdf5') and paths[i].startswith(mode_path)):
        i+= 1
    # number_rotors = np.load(os.path.join(root_path, paths[i])).shape[0]
    number_rotors = hdf5.get_shape(os.path.join(root_path, paths[i]))[0]
    assert number_rotors in [1,4], f"Error: got {number_rotors} rotors in {os.path.join(root_path, paths[i])}, but supported only 1,4"
    return number_rotors

def inject_phases(sound, phase_modulation, interpolation_mode='bilinear'):
    # originally it is 4,250, 1710
    sound = np.transpose(sound, (1,0,2))
    sound = np.expand_dims(sound, 1)
    # at the end it is 250,1,4,1710
    sound = torch.from_numpy(sound)

    phase_modulation = torch.from_numpy(phase_modulation)

    if sound.dtype == torch.float32:
        sound = sound.to(torch.float64)

    modulated_sound = _apply_modulation(sound, phase_modulation, interpolation_mode=interpolation_mode)
    modulated_sound = modulated_sound.cpu().detach().numpy()

    modulated_sound = np.squeeze(np.transpose(modulated_sound, (2,0,3,1)), axis=-1)

    return modulated_sound

def build_constant_offset_phase_modulations(offset, num_rotors, time_samples):
    offset = np.deg2rad(offset)
    phase_modulation = np.ones((num_rotors, time_samples))
    for i in range(num_rotors):
        phase_modulation[i] = phase_modulation[i]*offset*i
    return phase_modulation

def build_time_varying_sine_constant_offset(offset, num_rotors, time_samples, with_zeros=True):
    offset = np.deg2rad(offset)
    '''
    phase_modulation = np.expand_dims(np.linspace(0,2*np.pi, time_samples, dtype=np.float64), 0)
    phase_modulation = np.tile(phase_modulation, (num_rotors,1))
    #for i in range(num_rotors):
        #phase_modulation[i] = phase_modulation[i]+offset*i
    phase_modulation = np.sin(phase_modulation)
    phase_modulation_with_zero = np.zeros((num_rotors,time_samples+1))
    phase_modulation_with_zero[:, :time_samples//2] = phase_modulation[:, :time_samples//2]
    phase_modulation_with_zero[:, time_samples//2] = 0
    phase_modulation_with_zero[:, time_samples//2+1:] = phase_modulation[:,time_samples//2:]
    print(phase_modulation_with_zero.shape)
    '''
    phase_modulation = np.expand_dims(np.linspace(0,2*np.pi, time_samples), 0)
    phase_modulation = np.tile(phase_modulation, (num_rotors,1))
    for i in range(num_rotors):
        phase_modulation[i] = phase_modulation[i]+offset*i
    phase_modulation = np.sin(phase_modulation)
    if with_zeros:
        phase_modulation_with_zero = np.zeros((4,8*128+2))


        phase_modulation_with_zero[0, :512] = phase_modulation[0, :512]
        phase_modulation_with_zero[0, 512] = 0
        phase_modulation_with_zero[0, 513:-1] = phase_modulation[0,512:]
        phase_modulation_with_zero[2, 1:512] = phase_modulation[2, 1:512]
        phase_modulation_with_zero[2, 512] = 0
        phase_modulation_with_zero[2, 513:-1] = phase_modulation[2,512:]

        phase_modulation_with_zero[1, :256] = phase_modulation[1, :256]
        phase_modulation_with_zero[1, 256] = 0
        phase_modulation_with_zero[1, 257:769] = phase_modulation[1,256:768]
        phase_modulation_with_zero[1, 769] = 0
        phase_modulation_with_zero[1, 770:] = phase_modulation[1,768:]


        phase_modulation_with_zero[3, :256] = phase_modulation[3, :256]
        phase_modulation_with_zero[3, 256] = 0
        phase_modulation_with_zero[3, 257:769] = phase_modulation[3,256:768]
        phase_modulation_with_zero[3, 769] = 0
        phase_modulation_with_zero[3, 770:] = phase_modulation[3,768:]
        
        return phase_modulation_with_zero
    else:
        return phase_modulation


def numpy_get_shape(name:str):
    # gets shape of an array in file and returns its shape without loading it
    assert name.endswith('.npy'), "The name passed as input is not valid (it doesn't finish with .npy)"

    return np.load(name, mmap_mode='r').shape

def get_number_of_samples(directory: list, root_path:str):
    #return min(numpy_get_shape(os.path.join(root_path, x))[-1] for x in directory)
    return min(hdf5.get_shape(os.path.join(root_path, x))[-1] for x in directory if x.endswith(".hdf5"))

def create_video(path):
    FRAME_RATE:int = 30
    FONT = 0 #cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1
    FONT_COLOR = (0, 0, 0)  
    FONT_THICKNESS = 2

    image_folder = os.path.join(path, 'frames')
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key = lambda s: int(s.removeprefix("frame_").split(".png")[0]))

    # Get the dimensions of the first image to determine video frame size
    img = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = img.shape
    frame_size = (width, height)

    output_video = os.path.join(path, 'video_visualization.mp4')

    # Initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, FRAME_RATE, frame_size)

    for idx,image in tqdm(enumerate(images)):
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        title = f"Frame {idx}"
        cv2.putText(frame, title, (10, 30), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        video.write(frame)

    # Release the VideoWriter and close the video file
    video.release()

    # Optional: If you want to delete the individual PNG images after creating the video


    print(f"Video '{output_video}' has been created.")
        
if __name__ == '__main__':

    args = parse_arguments()

    root_path:str = os.path.join("/mnt", "walkure_public", "tamirs", "simulator", "raw_data", args['path'])
    #root_path: str = "/mnt/walkure_public/tamirs/simulator/raw_data/entire_room_org_2.5_2.5_rotors_4_free_space_and_indoor/"
    assert os.path.exists(root_path), f"Given the loading path {root_path}, but it doesn't exist! Please set a different --path!"

    

    if args['mode'] == 'free_space':
        MODE = 'images0_y_'
    elif args['mode'] == 'indoor':
        MODE = 'indoor_y_'
    else:
        raise ValueError('You need to set --mode to either free_space or indoor')

    directory = list(filter(lambda x: x.endswith('.hdf5') and x.startswith(MODE), os.listdir(root_path)))
    number_of_samples:int = get_number_of_samples(directory, root_path)
    directory = sorted(directory, key= lambda x: float(x.removeprefix(MODE).removesuffix('.hdf5')))

    NUM_ROTORS:int = get_number_rotors(root_path=root_path, mode_path=MODE)
    # START_REVOLUTION:int = 2
    # WINDOW_LEN:int = 128
    # NUM_REVOLUTIONS:int = 8
    
    # first_sample:int = WINDOW_LEN*START_REVOLUTION
    # last_sample:int = WINDOW_LEN * (START_REVOLUTION+NUM_REVOLUTIONS)
    # number_of_samples:int = last_sample-first_sample

    if args['phase_modulation'] == "constant_offset":
        phase_modulation = build_constant_offset_phase_modulations(90, NUM_ROTORS, number_of_samples)
        phase_modulation_path = args['phase_modulation']
    elif args['phase_modulation'] == "time_varying_sine":
        phase_modulation = build_time_varying_sine_constant_offset(90, NUM_ROTORS, number_of_samples)
        phase_modulation_path = args['phase_modulation']
    else:
        assert os.path.exists(args['phase_modulation']), f"The path that you provided ({args['phase_modulation']}) does not exist. Provide a path that exists or choose among constant_offset and time_varying_sine"
        phase_modulation = np.load(args['phase_modulation'])
        phase_modulation_path = args['phase_modulation'].removesuffix('.npy').split('/')[-1]
        print(f"phase modulation path: {phase_modulation_path}")

        number_of_samples = phase_modulation.shape[-1]

    assert len(phase_modulation.shape) == 2 and phase_modulation.shape[0] == NUM_ROTORS, f"The phase modulation is expected to have shape ({NUM_ROTORS}, time) but the one that you provided has shape {phase_modulation.shape}. Please load a phase modulation with the right shape"

    saving_root_path: str = os.path.join('/mnt', 'walkure_public', 'tamirs', 'simulator', 'processed_data')
    EXPERIMENT_NAME: str = args['path'] #'entire_room_org_2.5_2.5_rotors_4_free_space_and_indoor'
    saving_root_path = os.path.join(saving_root_path, EXPERIMENT_NAME, phase_modulation_path, f"{args['mode']}_")
    if not os.path.exists(saving_root_path):
        os.makedirs(saving_root_path)

    temporal_positions = np.arange(0,PLOT_DT*number_of_samples, PLOT_DT)
    np.save(os.path.join(saving_root_path, 'temporal_positions.npy'), temporal_positions)
    print(f"Temporal positions saved in {os.path.join(saving_root_path, 'temporal_positions.npy')}")

    rotor_position = build_rotor_position(num_rotors=NUM_ROTORS, number_of_samples=number_of_samples, phase_modulation=phase_modulation)
    np.save(os.path.join(saving_root_path, 'rotor_position.npy'), rotor_position)
    print(f"Rotor position saved in {os.path.join(saving_root_path, 'rotor_position.npy')}")

    x_axis = np.arange(0, 5.0, 0.02)
    np.save(os.path.join(saving_root_path, 'spatial_locations_x.npy'), x_axis)
    print(f"Spatial locations x saved in {os.path.join(saving_root_path, 'spatial_locations_x.npy')}")

    y_axis = np.arange(0, 5.0, 0.02)
    np.save(os.path.join(saving_root_path, 'spatial_locations_y.npy'), y_axis)
    print(f"Spatial locations y saved in {os.path.join(saving_root_path, 'spatial_locations_y.npy')}")

    MICS_R: float = 0.51
    mask, origin2D = build_mask(num_rotors=NUM_ROTORS, mics_R=MICS_R, org_x=args['org_x'], org_y=args['org_y'])
    rotor_coordinates_file_path: str = os.path.join(saving_root_path, 'rotors_coordinate.txt')
    with open(rotor_coordinates_file_path, "w") as f:
        f.write(f"The rotors are placed at {origin2D[:,0]} m with a radius of {MICS_R} m each")

    # work on the 8 microphones
    # if NUM_ROTORS == 4:
    #     EIGHT_MICROPHONES_LOADING_PATH: str = "/mnt/walkure_public/tamirs/pressure_field_2d_no_padding/indoor_recordings_4_rotors_8_mics_d_0.05_mode_indoor_None/"
    #     EIGHT_MICROPHONES_LOADING_PATH = os.path.join(EIGHT_MICROPHONES_LOADING_PATH, f"{round(float(args['org_x'])*100)}_{round(float(args['org_y'])*100)}.npy")
    #     eight_microphones = np.load(EIGHT_MICROPHONES_LOADING_PATH)
    #     eight_microphones = eight_microphones[..., :number_of_samples]
    #     eight_microphones_modulated = inject_phases(eight_microphones, phase_modulation, interpolation_mode='bicubic')
    #     eight_microphones_modulated = np.sum(eight_microphones_modulated, axis=0)
    #     eight_microphones_modulated_saving_path: str = os.path.join(saving_root_path, f'8_microphones_modulated_recordings.npy')
    #     np.save(eight_microphones_modulated_saving_path, eight_microphones_modulated)
    #     print(f'8 microphones modulated recording saved in {eight_microphones_modulated_saving_path}')
    # else:
    #     raise ValueError(f"Cannot work on 8 microphones since NUM_ROTORS {NUM_ROTORS} is not 4")

    grid = []
    cut = np.inf
    
    print(f"Iterating over {root_path} with mode {MODE} (length: {len(directory)})")
    print(directory)

    for yi, path in tqdm(enumerate(directory)):
        
        # pressure_field = np.load(os.path.join(root_path, path))
        pressure_field = hdf5.load_numpy(os.path.join(root_path, path))
        assert pressure_field.shape[0] == NUM_ROTORS, f"Attention the array has a shape {pressure_field.shape} while we expected it to have at the first axis a dimension of {NUM_ROTORS}"
        
        # Inject the phases here
        #pressure_field = pressure_field[..., first_sample:last_sample]

        pressure_field = inject_phases(pressure_field, phase_modulation, interpolation_mode='bilinear')

        # sum over rotor dimension
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
    image_saving_path:str = os.path.join(saving_root_path, 'frames')

    if not os.path.exists(image_saving_path):
        os.makedirs(image_saving_path)

    plt.ioff()
    #fig, ax = plt.subplots()
    for i in tqdm(range(grid_db.shape[-1])):
    #     ax.clear()
    #     ax.imshow(grid_db[..., i], extent=[0.0,5.0,0.0,5.0], vmin=vmin, vmax=vmax, origin='lower')
    #     ax.colorbar()
    #     ax.clim(vmin,vmax)
    #     fig.savefig(os.path.join(image_saving_path, f"frame_{i}.png"), dpi=200)

    # plt.close(fig)

        plt.imshow(grid_db[..., i], extent=[0.0,5.0,0.0,5.0], vmin=vmin, vmax=vmax, origin='lower')
        plt.colorbar()
        plt.clim(vmin, vmax)
        plt.savefig(os.path.join(image_saving_path, f"frame_{i}.png"), dpi=200)
        plt.clf()
        plt.close()

    print(f'Frames saved in {image_saving_path}')

    create_video(saving_root_path)