# BEFORE: from jax.api import jit
from jax import jit
import jax.numpy as jnp
import numpy as onp
from .jax_geometry import circular_2D_array, circular_2D_array_jax_jitted, linear_array_based_dist_from_org 
from .acoustic_methods import modified_simulate
#from .acoustic_methods_onp import image_source_model_onp, compute_rir_threaded 
from .acoustic_methods import image_source_model, compute_rir_threaded 
from .plot_room import plot2DRoom
from time import time

import pdb
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from data_processing.create_signals_manually import *


#################################################################
############### Handling rotor creation (sources) ###############
#################################################################

def create_circular_phased_rotor_array(origin2D, num_sources_in_circle, radiuses_circles, phi0=0, enable_prints=False):
    if enable_prints:
        print(f'Adding {radiuses_circles.shape[0]} circular source arrays with {num_sources_in_circle} sources each.')

    # drop sources on phased array positions
    sources_pos = None
    for i in range(radiuses_circles.shape[0]):
        R, _ = circular_2D_array_jax_jitted(center=origin2D[i], M=num_sources_in_circle, phi0=phi0, radius=radiuses_circles[i])
        if sources_pos is None:
            sources_pos = R
        else:
            sources_pos = jnp.hstack([sources_pos, R])
    
    return sources_pos


def create_linear_phased_rotor_array(origin2D, 
                            distances, phi_array, enable_prints=False):
    # distances are the distances of each source from the origin
    # phi_array is the angle of the array (with respect to x,y coords)
    if enable_prints:
        print(f'Adding a linear source array with {len(distances)} sources.')
    
    sources_pos, _ = linear_array_based_dist_from_org(center=origin2D, distances=distances, phi0=phi_array)
    
    return sources_pos


def add_sources_phased_rotor_array(origin2D,
                            omega, phies0, magnitudes, delays, num_sources_in_circle=None, radiuses_circles=None,
                            distances=None, phi_array=None, phi0_sources=0, harmonies=[1],
                            duration=0.5, fs=44100, enable_prints=False):
    
    if distances is not None:
        # add a linear phased array
        sources_pos = create_linear_phased_rotor_array(origin2D, 
                            distances, phi_array, enable_prints) #TODO: fix function (if wiil be needed...)
    else:
        # add a circular phased array
        sources_pos = create_circular_phased_rotor_array(origin2D, num_sources_in_circle, radiuses_circles, phi0_sources, enable_prints)

    # create signals
    signals = create_signals(omega=omega, phies0=phies0, magnitudes_in=magnitudes, duration=duration, 
                            fs=fs, num_sources_in_circle=num_sources_in_circle, num_radiuses_circles= radiuses_circles.shape[0], harmonies=harmonies)
    
    # create list of dicts of sources
    sources = []
    for i, pos in enumerate(sources_pos.T):
        source = {'pos': pos, 'signal': signals[i], 'images': [], 'delay': delays[i]}
        sources.append(source)

    return sources

def add_sources_phased_rotor_array_no_signals_jax(origin2D,
                            delays, num_sources_in_circle=None, radiuses_circles=None,
                            phi0_sources=0,
                            enable_prints=False):

    # add a circular phased array

    if enable_prints:
        print(f'Adding {radiuses_circles.shape[0]} circular source arrays with {num_sources_in_circle} sources each.')

    # drop sources on phased array positions
    sources_pos = None
    for i in range(radiuses_circles.shape[0]):
        R, _ = circular_2D_array_jax_jitted(origin2D[i], num_sources_in_circle, phi0_sources, radiuses_circles[i])
        if sources_pos is None:
            sources_pos = R
        else:
            sources_pos = jnp.hstack([sources_pos, R])
    
    # create list of dicts of sources
    sources = []
    for i, pos in enumerate(sources_pos.T):
        source = {'pos': pos, 'images': [], 'delay': delays[i]}
        sources.append(source)

    return sources

def add_sources_phased_rotor_array_no_signals(origin2D,
                            delays, num_sources_in_circle=None, radiuses_circles=None,
                            distances=None, phi_array=None, phi0_sources=0,
                            enable_prints=False):
    
    if distances is not None:
        # add a linear phased array
        sources_pos = create_linear_phased_rotor_array(origin2D, 
                            distances, phi_array, enable_prints) #TODO: fix function (if wiil be needed...)
    else:
        # add a circular phased array
        sources_pos = create_circular_phased_rotor_array(origin2D, num_sources_in_circle, radiuses_circles, phi0_sources, enable_prints)

    # create list of dicts of sources
    sources = []
    for i, pos in enumerate(sources_pos.T): #TODO ask Tom
        source = {'pos': pos, 'images': [], 'delay': delays[i]}
        #source = {'pos': pos, 'signal': signals[i], 'images': [], 'delay': delays[i]}

        sources.append(source)

    return sources

#################################################################
################# Handling microphones creation #################
#################################################################

def add_mics_to_room(origin2D, fs, mics_R=0.1, phi_mics=0, enable_prints=False, distances=None, coords=None, phi_delays=None, num_mics_circ_array=1,
                    grid_mics_y=-1, delta_mics=2, room_max_x=5., mics_around_rotor_num=0):
    # add microphone arrays   
    if grid_mics_y != -1:
        # GUESS: we are inserting microphones with a fixed y coordinate and a changing coordinate
        #        and we filter out the microphones which are inside a certain circle which
        #        contains the origin (filter out the microphones which are too close)
        R, phi = [], None
        x = onp.arange(0, room_max_x, delta_mics)
        
        for xi in x:
            xi_2 = round(xi, 2)
            if onp.any((xi_2-origin2D[:,0])**2 + (grid_mics_y-origin2D[:,1])**2 < mics_R**2):
                continue
            R.append([xi_2, grid_mics_y])

        R = onp.array(R).T
        phi = onp.zeros(len(R[0]))
        
    elif distances is not None: # create a linear mic array based on given distances from the rotor's origin
        org = origin2D[mics_around_rotor_num]
        R, phi = linear_array_based_dist_from_org(center=org, distances=distances, phi0=phi_mics)
    elif coords is not None: # create array based on coords
        R = onp.array(coords)
        phi = onp.zeros(len(R[0]))
    else: # create a circular array of mics
        R, phi = circular_2D_array(center=origin2D[0], M=num_mics_circ_array, phi0=phi_mics, radius=mics_R)
    
    if phi_delays is None:
        phi_delays = onp.zeros_like(phi)

    num_mics = R.shape[1]

    mics_array = {'M': num_mics, 'fs': fs, 'R': R, 'phi': phi_delays}

    if enable_prints:
        print(f'Added {num_mics} mics')

    return mics_array

def add_mics_to_room_jax(origin2D, fs, mics_R=0.1, phi_mics=0, enable_prints=False, num_mics_circ_array=1):
    # add microphone arrays    
    # create a circular array of mics
    R, phi = circular_2D_array_jax_jitted(origin2D[0], num_mics_circ_array, phi_mics, mics_R)
    
    phi_delays = onp.zeros_like(phi)

    num_mics = R.shape[1]

    mics_array = {'M': num_mics, 'fs': fs, 'R': R, 'phi': phi_delays}

    if enable_prints:
        print(f'Added {num_mics} mics')

    return mics_array

#################################################################
##################### Computing ISM and RIR #####################
#################################################################

def compute_ISM_and_RIR_phased_sources_array(origin2D, origin_mics,
                                            room, delays,
                                            mics_distances, phi_mics=0, coords_mics=None, num_mics_circ_array=1, radiuses_circles=None, num_sources_in_circle=None,
                                            sources_distances=None, phi_sources_array=None, phi0_sources=0,
                                            fs=44100, max_order=0, grid_mics_y=-1, delta_mics=2, mics_R=0.1,
                                            mics_around_rotor_num = 0,
                                            enable_prints=False, is_plot_room=False, room_plot_out=None):
    sources = add_sources_phased_rotor_array_no_signals(origin2D,
                            delays, num_sources_in_circle=num_sources_in_circle, radiuses_circles=radiuses_circles,
                            distances=sources_distances, phi_array=phi_sources_array, phi0_sources=phi0_sources,
                            enable_prints=enable_prints)
    
    # mics_array = add_mics_to_room_jax(origin_mics, fs=fs, distances=mics_distances, phi_mics = phi_mics, coords=coords_mics, mics_R=mics_R,
    #                             enable_prints=enable_prints, num_mics_circ_array=num_mics_circ_array, 
    #                             grid_mics_y=grid_mics_y, delta_mics=delta_mics, room_max_x=room['corners'][0][0],
    #                             mics_around_rotor_num=mics_around_rotor_num)
    mics_array = add_mics_to_room_jax(origin_mics, fs=fs, phi_mics = phi_mics, mics_R=mics_R,
                                enable_prints=enable_prints, num_mics_circ_array=num_mics_circ_array)
    
    # plot room
    if is_plot_room:
        plot2DRoom(room, mics_array, sources, img_order=0, marker_size=5, room_plot_out=room_plot_out)

    # compute ISM
    if enable_prints:
        print('Computing ISM', flush=True)

    if max_order == 0:
        # data is taken from pyRoomAcoustics' image source model
        visibility = [onp.ones((mics_array['M'], 1), dtype=onp.int32)] * len(sources)
        for source in sources:
            source['images'] = onp.expand_dims(source['pos'], axis=1)
            source['damping'] = onp.ones(1)
            source['generators'] = -onp.ones(1)
            source['walls'] = -onp.ones(1)
            source['orders'] = onp.zeros(1)
    else:
        #visibility, sources = image_source_model_onp(room, sources, mics_array, max_order=max_order)
        visibility, sources = image_source_model(room, sources, mics_array, max_order=max_order)
    # CONTINUE FROM HERE
    # compute RIR
    # start_time = time()
    if enable_prints:
        print('Computing RIR', flush=True)
    rir, rir_by_orders = compute_rir_threaded(sources, mics_array, visibility, room['fs'], room['t0'], max_order)
    # print(f'Ended computing rir, time: {time()-start_time}')
    return sources, mics_array, rir, rir_by_orders


#################################################################
################## simulating RIR with signals ##################
#################################################################


def create_signals(omega, phies0, magnitudes_in, duration=0.5, fs=44100,  num_sources_in_circle=4, num_radiuses_circles=1, harmonies=[1], modulate_phase=False):
    phies0_flattened = jnp.reshape(phies0, (num_radiuses_circles * num_sources_in_circle, harmonies.shape[0])) # Note! the order is: inner loop -> outer loop -> inner loop -> outer loop -> ...
    magnitudes_in_flattened = jnp.reshape(magnitudes_in, (num_radiuses_circles * num_sources_in_circle, harmonies.shape[0])) # Note! the order is: inner loop -> outer loop -> inner loop -> outer loop -> ...
    num_rotors = int(magnitudes_in.shape[0] / num_sources_in_circle)
    signals = create_harmonic_signal_matrix_style_with_coeff(omega, magnitudes_in_flattened.T, phi_0=phies0_flattened.T, duration=duration, harmonies=harmonies, fs=fs, modulate_phase=modulate_phase, num_rotors=num_rotors)
    return signals

# jit to accelarate the computation of signals creaation
create_signals_jitted = jit(create_signals, static_argnums=(0,3,4,5,6))


def create_signals_and_simulate_recordings(rir, num_mics, 
                                            fs, duration, omega, phies0, magnitudes, harmonies, 
                                            num_sources_in_circle, radiuses_circles, delay_sources, max_rec_len, flip_rotation_direction, modulate_phase=False, recordings_foreach_rotor=False):

    signals = create_signals(omega, phies0, magnitudes, duration, 
                            fs, num_sources_in_circle, radiuses_circles.shape[0], harmonies, modulate_phase)
    
    for i in range(radiuses_circles.shape[0]):
        if flip_rotation_direction[i//2]:
            signals = signals.at[num_sources_in_circle * i:num_sources_in_circle * (i+1)].set(signals[num_sources_in_circle * i:num_sources_in_circle * (i+1)][::-1])

    if recordings_foreach_rotor:
        recordings = []
        for i in range(radiuses_circles.shape[0]//2):
            cur_rotor_recordings = modified_simulate(rir[:, num_sources_in_circle * 2*i:num_sources_in_circle * (2*i + 2)], 
                                                 signals[num_sources_in_circle * 2*i:num_sources_in_circle * (2*i + 2)], 
                                                    delay_sources, num_sources_in_circle*2, num_mics, max_rec_len)
            
            recordings.append(cur_rotor_recordings)
        recordings = jnp.array(recordings) # shape: (#rotors, #mics, #sig_len)
    else:
        recordings = modified_simulate(rir, signals, delay_sources, num_sources_in_circle*radiuses_circles.shape[0], num_mics, max_rec_len)
    
    return recordings