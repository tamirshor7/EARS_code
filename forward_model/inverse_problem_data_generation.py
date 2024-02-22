from forward_model import main_forward, add_args
import os
import argparse
import numpy as onp
from tqdm import tqdm

def generate_datapoint(absorption_coeff:float,x: float, room_side: float = 120.0, num_microphones:int = 8):
    '''
    Generate a recording for each microphone in a (large) square room to simulate an infinitely long wall distant from the drone by x meters.
    :param x: distance from the drone to the wall
    :param room_side: side length of the room
    :param num_microphones: number of microphones in the room
    :return recordings: (num_microphones, t) simulated recording for each microphone, where t is the number of samples
    '''
    # Prepare arguments for the forward model
    args = []
    args.extend("-exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051".split())
    args.extend("-opt_harmonies 0.5 1 2 3".split())
    args.extend("-opt_harmonies_init_vals 0 0 0 0".split())
    args.extend("-opt_phi_0_init_vals 0 0 0 0".split())
    args.extend("-num_sources_power 7".split())
    args.extend("-channel 0".split())
    args.extend("-duration 0.5".split())
    args.extend("-num_rotors 4".split())
    args.extend(f"-e_absorption {absorption_coeff}".split())
    args.extend("-mics_rads 0.53".split())
    args.extend("-max_order 1".split())

    args.extend(f"-num_phases_mics {num_microphones}".split())
    args.extend(f"-room_x {room_side}".split())
    args.extend(f"-room_y {room_side}".split())
    args.extend(f"-org_x {room_side/2}".split())
    args.extend(f"-org_y {x}".split())

    # build the parser and parse the arguments
    parser = argparse.ArgumentParser()
    arguments = add_args(parser, args)
    
    # run the forward model
    recordings = main_forward(arguments)
    recordings = onp.asarray(recordings)

    return recordings

def generate_dataset(absorption_coeff:float,num_datapoints: int, min_distance:float = 1.0, max_distance:float = 5.0, room_side: float = 120.0, num_microphones:int = 8):
    '''
    Generate a dataset of num_datapoints recordings for each microphone in a (large) square room to simulate an infinitely long wall distant from the drone by x meters.
    :param num_datapoints: number of recordings to generate
    :param min_distance: minimum distance from the drone to the wall
    :param max_distance: maximum distance from the drone to the wall
    :param room_side: side length of the room
    :param num_microphones: number of microphones in the room
    :return recordings: (num_datapoints, num_microphones, t+1) simulated recording for each microphone, where t is the number of samples
                        ATTENTION: the last column stores the distance from the drone to the wall (x)
    '''
    recordings = None
    distances = onp.linspace(min_distance,max_distance,num_datapoints)#onp.random.uniform(low = min_distance, high=max_distance, size=num_datapoints)
    min_recording_length = float('inf')
    with tqdm(total=num_datapoints) as pbar:
        for x in distances:
            data_point = generate_datapoint(absorption_coeff,x, room_side, num_microphones)
            if recordings is None:
                recordings = data_point
                recordings = recordings.reshape((1, num_microphones, data_point.shape[1]))
                min_recording_length = data_point.shape[1]
                continue
            # make sure all recordings have the same length
            elif data_point.shape[1] < min_recording_length:
                min_recording_length = data_point.shape[1]
                recordings = recordings[:, :, :min_recording_length]
            else:
                data_point = data_point[:, :min_recording_length]

            data_point = data_point.reshape((1, num_microphones, min_recording_length))
            recordings = onp.concatenate((recordings, data_point), axis=0)
            
            pbar.update(1)
    # add the distance from the drone to the wall to the last column of the recordings
    distances = onp.repeat(distances, num_microphones)
    # insert distances as the last column of the recordings
    distances = distances.reshape((num_datapoints, num_microphones, 1))
    print(f"Shape of the distances: {distances.shape}")
    print(f"Shape of the recordings: {recordings.shape}")
    recordings = onp.concatenate((recordings, distances), axis=2)
    return recordings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-abs_coeff', default=0.2, type=float)
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args = get_args()
    absorption_coeff = args['abs_coeff']

    num_datapoints = 50
    output_dir = os.path.join(cur_dir, '..', '..', 'data',f"{absorption_coeff}", 'inverse_problem_data_wall_x_all_0_batch8')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    global_min = 1
    global_max = 5
    step_size = (global_max-global_min)/10

    for i in range(10):
        min_range = global_min+i*step_size
        max_range = min_range + step_size
        recordings = generate_dataset(absorption_coeff,num_datapoints,min_range,max_range)
        print("Shape of the recordings: ")
        print(recordings.shape)
        # Save the file with the right index for the name
        number = 0
        while os.path.exists(os.path.join(output_dir, f'inverse_problem_data_{number}.npy')):
            number += 1
        onp.save(os.path.join(output_dir, f'inverse_problem_data_{number}.npy'), recordings)
        print("Inverse problem data saved to: ", os.path.join(output_dir, f'room_inverse_problem_data_{i}.npy'))

