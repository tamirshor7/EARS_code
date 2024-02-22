import jax.numpy as np
from jax.config import config
config.update("jax_enable_x64", True)

from scipy.io import wavfile
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from pyroomacoustics_differential.room import from_corners, extrude
from pyroomacoustics_differential.acoustic_methods import image_source_model , compute_rir, simulate


def add_source_and_mics(signal, origin_source, origin_mic, fs):
    # add sources
    sources = []
    mics = []
    print(f'Adding {1} sources')
    
    #pos = np.expand_dims(np.asarray(origin_source),1)
    source = {'pos': np.asarray(origin_source), 'signal': signal, 'images': [], 'delay': 0.}
    sources.append(source)

    # add microphone arrays
    print(f'Adding a single mic')
    mics = {'M': 1, 'fs': fs, 'R': np.asarray(origin_mic)}

    return sources, mics


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES']= '3'

    # specify signal source
    fs, signal = wavfile.read(os.path.join('test_data', 'arctic_a0010.wav'))

    corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T  # [x,y]

    # set max_order to a low value for a quick (but less accurate) RIR
    room_2D = from_corners(corners, fs=fs, max_order=1, absorption=0.2)
    room_3D = extrude(room_2D, height=2.)

    # add source to 3D room
    sources, mics = add_source_and_mics(origin_source=[1., 1., 0.5], origin_mic=[[3.5, 3.6], [2., 2.], [0.5,  0.5]], signal=signal, fs=fs)

    # compute ISM
    print('Computing ISM')
    visibility, sources = image_source_model(room_3D, sources, mics)

    # compute RIR
    print('Computing RIR')
    rir = compute_rir(sources, mics, visibility, room_3D['fs'], room_3D['t0'])
    
    # simulate
    print('Simulating recordings')
    recordings = simulate(rir, sources, mics, room_fs=room_3D['fs'])
    print(recordings[0])
