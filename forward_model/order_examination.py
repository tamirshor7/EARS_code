#%%
import os, sys
import itertools
from multiprocessing import Pool

import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from pyroomacoustics_differential.acoustic_methods_onp import image_source_model_onp, compute_rir_threaded
from pyroomacoustics_differential.room import from_corners
from pyroomacoustics_differential.consts import inch_to_meter

#%%

def parse_source_mic_pos(source_mic_pos):
    source_pos, mic_pos = source_mic_pos
    source_x = onp.round(source_pos[0],2)
    source_y = onp.round(source_pos[1],2)
    mic_x = onp.round(mic_pos[0],2)
    mic_y = onp.round(mic_pos[1],2)
    return source_x, source_y, mic_x, mic_y


def process_postion(pos_params):
    source_mic_pos, fs, room2D, max_order, save_rir_dir = pos_params
    source_x, source_y, mic_x, mic_y = parse_source_mic_pos(source_mic_pos)

    if (source_x, source_y) == (mic_x, mic_y):
        return
    
    print(f'S({source_x},{source_y}), M({mic_x},{mic_y})')

    source_pos = onp.array((source_x, source_y))
    source = {'pos': source_pos, 'images': [], 'delay': 0}

    mics_pos = onp.array([[mic_x],[mic_y]])
    mics_array = {'M':1, 'fs':fs, 'R':mics_pos, 'phi':onp.array([0])}

    visibility, sources = image_source_model_onp(room2D, [source], mics_array, max_order=max_order)

    _, rir_by_orders = compute_rir_threaded(sources, mics_array, visibility, room2D['fs'], room2D['t0'], max_order)
    for order in range(max_order+1):
        cur_rir = rir_by_orders[order][0][0]
        save_path = os.path.join(save_rir_dir, f'order_{order}_source_{source_x}_{source_y}_mic_{mic_x}_{mic_y}.npy')
        onp.save(save_path, cur_rir)


def collect_rir(rir_dir, max_order, source_mic_positions):
    room_x = 5.
    room_y = 5.
    
    e_absorption = 0.2
    
    fs = 3003.735603735763

    corners = onp.array([[0,0], [0, room_y], [room_x, room_y], [room_x,0]]).T  # [x,y]
    room2D = from_corners(corners, fs=fs, e_absorption=e_absorption) # create room

    poses_params = [(source_mic_pos, fs, room2D, max_order, rir_dir) for source_mic_pos in source_mic_positions]

    with Pool(40) as pool:
        pool.map(process_postion, poses_params)



def create_diff_by_rir_orders(rir_dir, max_order, positions, epsilon=0.1):
    orders_by_diffs = {}
    for source_pos in positions:
        
        # process position of source with its mics
        source_x = onp.round(source_pos[0],2)
        source_y = onp.round(source_pos[1],2)
        
        print(f'S({source_x},{source_y})')

        orders_by_diffs[(source_x, source_y)] = []

        for mic_pos in positions:
            mic_x = onp.round(mic_pos[0],2)
            mic_y = onp.round(mic_pos[1],2)

            if (source_x, source_y) == (mic_x, mic_y):
                orders_by_diffs[(source_x, source_y)].append(-1)
                continue
            
            rir_path = os.path.join(rir_dir, f'order_0_source_{source_x}_{source_y}_mic_{mic_x}_{mic_y}.npy')
            images_N = onp.load(rir_path)

            diffs = []

            for order in range(1, max_order+1):
                rir_path = os.path.join(rir_dir, f'order_{order}_source_{source_x}_{source_y}_mic_{mic_x}_{mic_y}.npy')
                images_i = onp.load(rir_path)
                diff = onp.linalg.norm(images_i) / onp.linalg.norm(images_N)
                images_N += images_i
                diffs.append(diff)
            min_order = onp.where(onp.asarray(diffs) < epsilon)[0].min() + 1
            orders_by_diffs[(source_x, source_y)].append(min_order)
    # onp.max([*orders_by_diffs.values()])
    return orders_by_diffs
    # df = pd.DataFrame.from_dict(orders_by_diffs, orient='index', columns=list(positions))
    # df.to_csv('test_df.csv')


# if __name__ == "__main__":
rir_dir = '/mnt/walkure_public/tomh/rir/rir_order_examination'
max_order = 15
range_positions = onp.arange(0.1, 2.7, 0.5)
positions = list(itertools.product(range_positions, range_positions))
source_mic_positions = itertools.product(positions, positions)

#collect_rir(rir_dir, max_order, source_mic_positions)

orders_by_diffs = create_diff_by_rir_orders(rir_dir, max_order, positions)

#%%
s_pos = positions[0]
rotor_rad = inch_to_meter(18)
rotor_2nd_rad = 0.51
delta_rad = 1.5 * rotor_rad / 2
for m_pos, order in zip(positions, orders_by_diffs[s_pos]):
    plt.scatter(m_pos[0], m_pos[1], marker=f'${order}$', c='k', linewidth=0.5)
plt.scatter(s_pos[0], s_pos[1])
plt.plot([0,5],[0,0], c='k')
plt.plot([0,5],[5,5], c='k')
plt.plot([0,0],[0,5], c='k')
plt.plot([5,5],[0,5], c='k')

circ = patches.Circle((s_pos[0] - delta_rad, s_pos[1]), rotor_rad, edgecolor='red', facecolor="None")
circ2 = patches.Circle((s_pos[0] + delta_rad, s_pos[1]), rotor_rad, edgecolor='red', facecolor="None")
plt.gca().add_patch(circ)
plt.gca().add_patch(circ2)


circ = patches.Circle((s_pos[0], s_pos[1]), rotor_rad + delta_rad, edgecolor='red', facecolor="None")
plt.gca().add_patch(circ)


plt.gca().set_aspect('equal', adjustable='box')


# %%
max_ordering = -1
max_rad = rotor_rad + delta_rad
for s_pos in positions:
    for m_pos, order in zip(positions, orders_by_diffs[s_pos]):
        if onp.sqrt((s_pos[0] - m_pos[0])**2 + (s_pos[1] - m_pos[1])**2) <= max_rad:
            max_ordering = max(max_ordering, order)

print(f'rad {max_rad}, max order {max_ordering}')
# %%
# for epsilon = 0.1 the max order is 6 with the radius of `max_rad = rotor_rad + delta_rad`