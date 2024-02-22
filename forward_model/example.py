#%%
import numpy as onp
import os, sys
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import chirp

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from pyroomacoustics_differential.acoustic_methods_onp import image_source_model_onp, compute_rir_threaded, simulate_onp
from pyroomacoustics_differential.room import from_corners
from pyroomacoustics_differential.plot_room import plot2DRoom
from pyroomacoustics_differential.forward_model_2D_interface import add_mics_to_room

# build room
room_x = 5.
room_y = 5.
max_order = 6
fs = 3003.735603735763
# fs, signal = wavfile.read(os.path.join("..","original_pyroomacoustics","pyroomacoustics","notebooks","birds.wav"))
# signal = signal[:fs*2,0]
duration = 3.
t = onp.linspace(0, duration, int(duration*fs))
#signal = chirp(t, f0=150, f1=10, t1=duration, method='linear')
# signal = onp.load('emitted_signals.npy')[150]

corners = onp.array([[0,0], [0, room_y], [room_x, room_y], [room_x,0]]).T  # [x,y]

# acoustics methods
#rt60 = 1.5#
#absorption, max_order, e_abs = inverse_sabine(rt60, [room_x, room_y])

#%%
source_pos = onp.array((2.2, 2.))# onp.load('source_240_pos.npy')
# source_pos2 = onp.array((3., 3.))# onp.load('source_240_pos.npy')
#mics_pos = onp.array([[source_pos[0]-1.],[source_pos[1]-2.5]])
# mics_pos = onp.array([[1., 1.],[1., 1.2]])
mics_pos = onp.array([[4.5],[3.1]])
#%%
room2D = from_corners(corners, fs=fs, e_absorption=0.2) # create room

# add source
# source_pos = onp.array([.5, .5])
signal = onp.sin(onp.linspace(-onp.pi, onp.pi, int(2 * fs)))
source = {'pos': source_pos, 'signal':signal, 'images': [], 'delay': 0}
# source2 = {'pos': source_pos2, 'signal':signal, 'images': [], 'delay': 0}

# add mics
# mics_array = add_mics_to_room([[0.93],[2.5]], mics_R=0.1, fs=fs, num_mics_circ_array=8)
mics_array = {'M':2, 'fs':fs, 'R':mics_pos, 'phi':onp.array([0])}


# compute ISM
time_start = time.time()
visibility, sources = image_source_model_onp(room2D, [source], mics_array, max_order=max_order)
print(f'ISM: {time.time()-time_start}')
# plot room
plot2DRoom(room2D, mics_array, sources, img_order=max_order, marker_size=24, room_plot_out=f'tmp_room{max_order}.png')

#%%
# compute RIR
time_start = time.time()
rir, rir_by_orders = compute_rir_threaded(sources, mics_array, visibility, room2D['fs'], room2D['t0'], max_order)
print(f'RIR: {time.time()-time_start}')

#%%
simulated_recordings_by_order = []
cur_rir = None
for i in range(max_order+1):
    cur_rir = rir_by_orders[i][0][0]
    plt.plot(cur_rir, label=i)
    #x = simulate_onp(cur_rir, sources, mics_array, fs)[0]
    x = rir_by_orders[i][0][0]
    simulated_recordings_by_order.append(x)
plt.title(f'RIR: Source ({round(source_pos[0],2)},{round(source_pos[1],2)}), Mic ({round(mics_pos[0][0],2)},{round(mics_pos[1][0],2)})')
plt.legend(loc='upper right')
plt.show()
plt.imsave()
# # %%
# diffs = []
# images_N = None
# epsilon = 0.1
# for i in range(max_order+1):
#     images_i = simulated_recordings_by_order[i][:6000].copy()
#     #images_i = onp.cumsum(images_i**2)
#     if i==0:
#         images_N = images_i
#     else:
#         images_N += images_i
#     diff = onp.linalg.norm(images_i) / onp.linalg.norm(images_N)
#     #diff = images_i[-1] / images_N[-1]
    
#     diffs.append(diff)

# plt.scatter(range(max_order+1), diffs)
# plt.grid()
# plt.xlabel('order')
# #plt.plot(onp.ones(max_order+1)*0.1*max(diffs), color='r', lw=0.7, label='10% max diff')
# plt.plot(onp.ones(max_order+1)*epsilon, color='r', lw=0.7, label=rf'$\varepsilon = {epsilon}$')
# plt.legend(loc='upper right')
# plt.title(f'Source location ({source_pos[0]},{source_pos[1]})    '+r'$\frac{||p_{i}||}{||\mathbf{P_i}||} \leq \varepsilon$')

# onp.where(onp.asarray(diffs) < epsilon)
#%%
diffs = []
epsilon = 0.1
images_N = simulated_recordings_by_order[0][:6000].copy()
for i in range(1, max_order+1):
    images_i = simulated_recordings_by_order[i][:6000].copy()
    diff = onp.linalg.norm(images_i) / onp.linalg.norm(images_N)
    images_N += images_i
    diffs.append(diff)

#plt.scatter(range(1,max_order+1), diffs)
plt.grid()
plt.xlabel('order')
#plt.plot(onp.ones(max_order+1)*0.1*max(diffs), color='r', lw=0.7, label='10% max diff')
#plt.plot(onp.ones(max_order+1)*epsilon, color='r', lw=0.7, label=rf'$\varepsilon = {epsilon}$')
plt.legend(loc='upper right')
plt.title(f'Source location ({source_pos[0]},{source_pos[1]})    '+r'$\frac{||p_{i+1}||}{||\mathbf{P_i}||} \leq \varepsilon$')

onp.where(onp.asarray(diffs) < epsilon)
plt.show()
# %%

xvalues = onp.arange(0.1, 2.7, 0.5)
yvalues = onp.arange(0.1, 2.7, 0.5)

xx, yy = onp.meshgrid(xvalues, yvalues)
counter = 0
from itertools import product
for r in product(xvalues, yvalues):
    plt.scatter(r[0], r[1])
    counter+=1
counter

# %%
