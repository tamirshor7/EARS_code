import os
import numpy as np
from tqdm import tqdm

# load direct component
cur_dir = os.path.dirname(os.path.abspath(__file__))
direct_component_path = os.path.join(cur_dir, '..', '..', 'data', 'rir', 'direct_component', 'direct_component.npy')
direct_component = np.load(direct_component_path)
print(f"Direct component shape: {direct_component.shape}")

# constants
ROOM_SIDE_MIN: float = 0.93
DELTA_STEP: float = 0.05
NUMBER_OF_DATAPOINTS_PER_AXIS: int = 63

def extract_first_order_component(complete_rir):
    padding_size = ((0,0),(0,0),(0, complete_rir.shape[-1] - direct_component.shape[-1]))
    direct = np.pad(direct_component, padding_size)
    return complete_rir-direct

def get_index(filename):
    # extract the coordinates from the filename 
    x, y =  (int(filename.strip().split('.')[0].split('_')[0]), int(filename.strip().split('.')[0].split('_')[1]))
    x = round(((x/100)-ROOM_SIDE_MIN)/DELTA_STEP)
    y = round(((y/100)-ROOM_SIDE_MIN)/DELTA_STEP)
    return x, y

if __name__ == '__main__':
    # obtained by:
    # mics_R = 0.9144
    # margins = 0.02
    # args['room_x'] = 5.0
    # delta = 0.05
    # np.arange( round((mics_R+margins), 2), round((args['room_x']-mics_R-margins),2), delta)
    matrix = np.zeros((NUMBER_OF_DATAPOINTS_PER_AXIS, NUMBER_OF_DATAPOINTS_PER_AXIS))

    complete_rir_path = os.path.join(cur_dir, '..', '..', 'data', 'rir', 'rir_indoor_4_channels_5.0_5.0_order_1_0.5_d_0.05','rir_indoor')
    d = {}
    for file in tqdm(set(os.listdir(complete_rir_path))):
        complete_rir = np.load(os.path.join(complete_rir_path, file))
        first_order_component = extract_first_order_component(complete_rir)
        x, y = get_index(file)
        assert matrix[x,y] == 0, f"This cell {x,y} has already been assigned! ({matrix[x,y]}) ({file}) ({d[(x,y)]})"
        d[(x,y)] = file
        matrix[x,y] = np.linalg.norm(first_order_component)/np.linalg.norm(complete_rir)
    
    # save matrix in file
    save_path = os.path.join(cur_dir, '..', '..', 'data', 'rir', 'relative_magnitude.npy')
    np.save(save_path, matrix)
    print(f"Matrix saved in {save_path}")
        