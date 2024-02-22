# BEFORE: from jax.api import jit
import torch
from jax import jit
import jax.numpy as jnp
import numpy as onp
from EARS.pyroomacoustics_differential.geometry import circular_2D_array, circular_2D_array_jax_jitted, linear_array_based_dist_from_org
from multibatch_acoustic_methods import torch_mb_modified_simulate as modified_simulate
from multibatch_acoustic_methods import torch_mb_modified_simulate_single_microphone, torch_mb_modified_simulate_multi_microphones
from EARS.pyroomacoustics_differential.acoustic_methods_onp import image_source_model_onp, compute_rir_threaded
from EARS.pyroomacoustics_differential.plot_room import plot2DRoom
from time import time

import pdb
import os, sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir, '..')
sys.path.insert(0, parent_dir)
from data_processing.create_signals_manually import *


#################################################################
############### Handling rotor creation (sources) ###############
#################################################################

def multibatch_create_harmonic_signal_matrix_style_with_coeff(omega, coeffs, phi_0=0, harmonies=[1], fs=44100, duration=0.5,
                                                   modulate_phase=None, num_rotors=1):
    fs = fs.min().item()
    signals_len = int(duration.item() * fs)
    t = jnp.expand_dims(jnp.linspace(0, duration, signals_len), 0)

    harmonies = jnp.array(harmonies)
    omega = omega.numpy()

    if len(coeffs) != len(harmonies):
        raise ValueError(f'Unmatching dims of coeffs. coeffs {len(coeffs)} vs harmonies {len(harmonies)}')

    harmonies = harmonies[:, jnp.newaxis]
    harmonies_omega_t = jnp.array([jnp.tile(harmonies * 2 * jnp.pi * omega[i] * t, (phi_0.shape[1], 1, 1)) for i in range(len(omega))])#jnp.tile(harmonies * 2 * jnp.pi * omega * t, (phi_0.shape[1], 1, 1))

    if modulate_phase is None:
        cos_res = jnp.cos(harmonies_omega_t + jnp.expand_dims(phi_0.T, 2))
    else:
        magic_number = 5 * 2  # revolutions #TODO: times 2 because we want to not count the first x revolutions
        spin_frac_of_sig = (2 * fs * magic_number / omega).astype(jnp.int32)
        num_sources = harmonies_omega_t.shape[1] // num_rotors
        # phase_modulation = jnp.tile(jnp.linspace(0, jnp.pi, spin_frac_of_sig), int(signals_len/spin_frac_of_sig)+1)[:signals_len]
        phase_modulation = jnp.array([jnp.tile(jnp.linspace(-jnp.pi, jnp.pi, spin_frac_of_sig[i]),
                                    int(signals_len / spin_frac_of_sig[i]) + 1)[:signals_len] for i in range(spin_frac_of_sig.shape[0])])#jnp.tile(jnp.linspace(-jnp.pi, jnp.pi, spin_frac_of_sig),
                                    #int(signals_len / spin_frac_of_sig) + 1)[:signals_len]  # TODO: from -pi to pi
        phase_modulation_single_rotor = jnp.tile(phase_modulation, (num_sources, 1, 1))
        no_modulation_single_rotor = jnp.zeros((num_sources, 1, harmonies_omega_t.shape[-1]))
        phase_modulation = jnp.zeros_like(harmonies_omega_t)
        # ATTENTION: this if-else is not in the original code and it is added to support the case when modulate_phase is not a list
        if num_rotors == 1:
            if modulate_phase:
                phase_modulation = phase_modulation.at[:,0:num_sources, :, :].set(phase_modulation_single_rotor)
            else:
                phase_modulation = phase_modulation.at[:,0:num_sources, :, :].set(no_modulation_single_rotor)
        else:
            for i in range(num_rotors):
                print(modulate_phase)
                if modulate_phase[i]:
                    phase_modulation = phase_modulation.at[i * num_sources:(i + 1) * num_sources, :, :].set(
                        phase_modulation_single_rotor)
                else:
                    phase_modulation = phase_modulation.at[i * num_sources:(i + 1) * num_sources, :, :].set(
                        no_modulation_single_rotor)
        cos_res = jnp.cos(harmonies_omega_t + jnp.expand_dims(phi_0.transpose(2,1,0), -1) + phase_modulation)

    samples = jnp.squeeze((jnp.expand_dims(coeffs.transpose(2,1,0), 2) @ cos_res), 2)
    return samples

def multibatch_torch_create_harmonic_signal_matrix_style_with_coeff(omega, coeffs, phi_0=0, harmonies=[1], fs=44100, duration=0.5,
                                                   modulate_phase=None, num_rotors=1):
    # coeffs: (batch_size, num_radiuses_circles*num_sources, num_harmonics) (num_rotors=1)
    # phi_0:  (batch_size, num_radiuses_circles*num_sources, num_harmonics) (num_rotors=1)
    # in our case modulate_phase = False
    device = fs.device
    # ATTENTION: why do we use min?! How does it make sense?
    fs = fs.min().item()
    signals_len = int(duration.item() * fs)
    # In this way t has shape (1, signals_len)
    t = torch.linspace(0, duration.item(), signals_len).unsqueeze(0).to(device)
    harmonies = torch.tensor(harmonies, device=device)

    if coeffs.shape[-1] != len(harmonies):
        raise ValueError(f'Unmatching dims of coeffs. coeffs {len(coeffs)} vs harmonies {len(harmonies)}')
    # with the following operation harmonies has shape (num_harmonics, 1)
    harmonies = harmonies.unsqueeze(-1)
    # shape of phi_0: (batch_size, num_radiuses_circles*num_sources, num_harmonics)
    # shape of t_repeat: (num_radiuses_circles*num_sources, 1, signals_len)
    # shape of harmonies_omega_t: (batch_size,num_radiuses_circles*num_sources, num_harmonics, signals_len) ((in our case num_rotors=1))
    harmonies_omega_t = torch.stack([harmonies * 2 * jnp.pi * omega[i] * t.repeat((phi_0.shape[1], 1, 1)) for i in range(omega.shape[0])],dim=0)
    #print(f'harmonies_omega_t has shape {harmonies_omega_t.shape}')
    if modulate_phase is None:
        raise NotImplementedError("Need to implement this for torch - cos_res = jnp.cos(harmonies_omega_t + jnp.expand_dims(phi_0.T, 2))")

    else:
        magic_number = 5 * 2  # revolutions #TODO: times 2 because we want to not count the first x revolutions
        spin_frac_of_sig = (2 * fs * magic_number / omega).to(torch.int32)
        num_sources = harmonies_omega_t.shape[1] // num_rotors
        # phase_modulation = jnp.tile(jnp.linspace(0, jnp.pi, spin_frac_of_sig), int(signals_len/spin_frac_of_sig)+1)[:signals_len]
        phase_modulation = torch.stack([
            # the modulation is performed in the following way:
            # first we compute a vector in range [-pi, pi] with length spin_frac_of_sig[i]
            # then we repeat this vector int(signals_len / spin_frac_of_sig[i]) + 1 times
            # this will make sure that the vector is long enough to cover the whole signal + a bit more
            # then we cut this vector by making sure that it has the same length as the signal
            torch.linspace(-torch.pi, torch.pi, spin_frac_of_sig[i]).repeat(int(signals_len / spin_frac_of_sig[i]) + 1)[:signals_len] 
            for i in range(spin_frac_of_sig.shape[0])])
        phase_modulation_single_rotor = phase_modulation.repeat(num_sources, 1, 1)
        no_modulation_single_rotor = torch.zeros((num_sources, 1, harmonies_omega_t.shape[-1]))
        phase_modulation = torch.zeros_like(harmonies_omega_t)
        # ATTENTION: this if-else is not in the original code and it is added to support the case when modulate_phase is not a list
        if num_rotors == 1:
            if modulate_phase:
                raise NotImplementedError(
                    "We didn't implement this for torch yet. In particular we need to take care of not doing in-place operations! (Otherwise we can't backpropagate!")
                # ATTENTION: We can't do in-place operations! Otherwise we can't backpropagate!
                phase_modulation[:,0:num_sources, :, :] = phase_modulation_single_rotor

            # In our case the else branch would yield the same result as not doing anything,
            # but given that there was an in-place operation (which would not have changed the result)
            # we need to skip it otherwise backpropagation will not work!
            #else:
                
                # ATTENTION: We can't do in-place operations! Otherwise we can't backpropagate!
                #phase_modulation[:,0:num_sources, :, :]=no_modulation_single_rotor
        else:
            raise NotImplementedError("We didn't implement this for torch yet. In particular we need to take care of not doing in-place operations! (Otherwise we can't backpropagate!")
            for i in range(num_rotors):
                print(modulate_phase)
                if modulate_phase[i]:
                    phase_modulation = phase_modulation.at[i * num_sources:(i + 1) * num_sources, :, :].set(
                        phase_modulation_single_rotor)
                else:
                    phase_modulation = phase_modulation.at[i * num_sources:(i + 1) * num_sources, :, :].set(
                        no_modulation_single_rotor)
        # By passing phi_0_flattened as phi_0 instead of phi_0_flattened.T we do not need to call permute!
        # cos_res = torch.cos(harmonies_omega_t + torch.unsqueeze(phi_0.permute(2,1,0), -1) + phase_modulation)

        # cos_res shape: (batch_size, num_radiuses_circles*num_sources, num_harmonics, num_samples)
        # phase_modulation shape: (batch_size, num_radiuses_circles*num_sources, num_harmonics, num_samples)
        cos_res = torch.cos(harmonies_omega_t + torch.unsqueeze(phi_0, -1) + phase_modulation)
    # By passing magnitudes_flattened as coeffs instead of magnitudes_flattened.T we do not need to call permute!
    # samples = torch.squeeze((torch.unsqueeze(coeffs.permute(2,1,0), 2).to(torch.double) @ cos_res), 2)
    
    # samples shape: (batch_size, num_radiuses_circles*num_sources, num_samples)
    samples = torch.squeeze((torch.unsqueeze(coeffs, 2).to(torch.double) @ cos_res), 2)
    return samples

# %%


def create_circular_phased_rotor_array(origin2D, num_sources_in_circle, radiuses_circles, phi0=0, enable_prints=False):
    if enable_prints:
        print(f'Adding {radiuses_circles.shape[0]} circular source arrays with {num_sources_in_circle} sources each.')

    # drop sources on phased array positions
    sources_pos = None
    for i in range(radiuses_circles.shape[0]):
        R, _ = circular_2D_array(center=origin2D[i], M=num_sources_in_circle, phi0=phi0, radius=radiuses_circles[i])
        if sources_pos is None:
            sources_pos = R
        else:
            sources_pos = onp.hstack([sources_pos, R])

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
                                                       distances, phi_array,
                                                       enable_prints)  # TODO: fix function (if wiil be needed...)
    else:
        # add a circular phased array
        sources_pos = create_circular_phased_rotor_array(origin2D, num_sources_in_circle, radiuses_circles,
                                                         phi0_sources, enable_prints)

    # create signals
    signals = create_signals(omega=omega, phies0=phies0, magnitudes_in=magnitudes, duration=duration,
                             fs=fs, num_sources_in_circle=num_sources_in_circle,
                             num_radiuses_circles=radiuses_circles.shape[0], harmonies=harmonies)

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
                                                       distances, phi_array,
                                                       enable_prints)  # TODO: fix function (if wiil be needed...)
    else:
        # add a circular phased array
        sources_pos = create_circular_phased_rotor_array(origin2D, num_sources_in_circle, radiuses_circles,
                                                         phi0_sources, enable_prints)

    # create list of dicts of sources
    sources = []
    for i, pos in enumerate(sources_pos.T):  # TODO ask Tom
        source = {'pos': pos, 'images': [], 'delay': delays[i]}
        # source = {'pos': pos, 'signal': signals[i], 'images': [], 'delay': delays[i]}

        sources.append(source)

    return sources


#################################################################
################# Handling microphones creation #################
#################################################################

def add_mics_to_room(origin2D, fs, mics_R=0.1, phi_mics=0, enable_prints=False, distances=None, coords=None,
                     phi_delays=None, num_mics_circ_array=1,
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
            if onp.any((xi_2 - origin2D[:, 0]) ** 2 + (grid_mics_y - origin2D[:, 1]) ** 2 < mics_R ** 2):
                continue
            R.append([xi_2, grid_mics_y])

        R = onp.array(R).T
        phi = onp.zeros(len(R[0]))

    elif distances is not None:  # create a linear mic array based on given distances from the rotor's origin
        org = origin2D[mics_around_rotor_num]
        R, phi = linear_array_based_dist_from_org(center=org, distances=distances, phi0=phi_mics)
    elif coords is not None:  # create array based on coords
        R = onp.array(coords)
        phi = onp.zeros(len(R[0]))
    else:  # create a circular array of mics
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
                                             mics_distances, phi_mics=0, coords_mics=None, num_mics_circ_array=1,
                                             radiuses_circles=None, num_sources_in_circle=None,
                                             sources_distances=None, phi_sources_array=None, phi0_sources=0,
                                             fs=44100, max_order=0, grid_mics_y=-1, delta_mics=2, mics_R=0.1,
                                             mics_around_rotor_num=0,
                                             enable_prints=False, is_plot_room=False, room_plot_out=None):
    sources = add_sources_phased_rotor_array_no_signals(origin2D,
                                                        delays, num_sources_in_circle=num_sources_in_circle,
                                                        radiuses_circles=radiuses_circles,
                                                        distances=sources_distances, phi_array=phi_sources_array,
                                                        phi0_sources=phi0_sources,
                                                        enable_prints=enable_prints)

    mics_array = add_mics_to_room(origin_mics, fs=fs, distances=mics_distances, phi_mics=phi_mics, coords=coords_mics,
                                  mics_R=mics_R,
                                  enable_prints=enable_prints, num_mics_circ_array=num_mics_circ_array,
                                  grid_mics_y=grid_mics_y, delta_mics=delta_mics, room_max_x=room['corners'][0][0],
                                  mics_around_rotor_num=mics_around_rotor_num)

    # plot room
    if is_plot_room:
        plot2DRoom(room, mics_array, sources, img_order=0, marker_size=5, room_plot_out=room_plot_out)

    # compute ISM
    if enable_prints:
        print('Computing ISM')

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
        visibility, sources = image_source_model_onp(room, sources, mics_array, max_order=max_order)
    # CONTINUE FROM HERE
    # compute RIR
    # start_time = time()
    if enable_prints:
        print('Computing RIR')
    rir, rir_by_orders = compute_rir_threaded(sources, mics_array, visibility, room['fs'], room['t0'], max_order)
    # print(f'Ended computing rir, time: {time()-start_time}')
    return sources, mics_array, rir, rir_by_orders


#################################################################
################## simulating RIR with signals ##################
#################################################################


def create_signals_torch(omega, phies0, magnitudes_in, duration=0.5, fs=44100, num_sources_in_circle=4,
                   num_radiuses_circles=1, harmonies=[1], modulate_phase=False):
    phies0_flattened = phies0.view((phies0.shape[0], num_radiuses_circles * num_sources_in_circle, harmonies.shape[0]))
    magnitudes_in_flattened = magnitudes_in.view((magnitudes_in.shape[0],num_radiuses_circles * num_sources_in_circle, harmonies.shape[0]))  # Note! the order is: inner loop -> outer loop -> inner loop -> outer loop -> ...
    num_rotors = int(magnitudes_in.shape[1] / num_sources_in_circle)
    # We don't need to transpose magnitudes_in_flattened and phies0_flattened because we already have the correct order! (indeed if we transpose them in 
    # multibatch_torch_create_harmonic_signal_matrix_style_with_coeff we need to transpose them again!)
    # signals = multibatch_torch_create_harmonic_signal_matrix_style_with_coeff(omega, magnitudes_in_flattened.T, phi_0=phies0_flattened.T,
    #                                                          duration=duration, harmonies=harmonies, fs=fs,
    #                                                          modulate_phase=modulate_phase, num_rotors=num_rotors)
    #print(f'inside create_signals_torch: omega {omega}, coeffs {magnitudes_in_flattened.shape}, phi_0 {phies0_flattened.shape}, duration {duration}, harmonies {harmonies}, fs {fs}, modulate_phase {modulate_phase}, num_rotors {num_rotors}')
    signals = multibatch_torch_create_harmonic_signal_matrix_style_with_coeff(omega=omega,coeffs=magnitudes_in_flattened, phi_0=phies0_flattened,
                                                             duration=duration, harmonies=harmonies, fs=fs,
                                                             modulate_phase=modulate_phase, num_rotors=num_rotors)
    return signals


def create_signals(omega, phies0, magnitudes_in, duration=0.5, fs=44100, num_sources_in_circle=4,
                   num_radiuses_circles=1, harmonies=[1], modulate_phase=False):
    phies0_flattened = jnp.reshape(phies0, (phies0.shape[0],num_radiuses_circles * num_sources_in_circle, harmonies.shape[
        0]))  # Note! the order is: inner loop -> outer loop -> inner loop -> outer loop -> ...
    magnitudes_in_flattened = jnp.reshape(magnitudes_in, (magnitudes_in.shape[0],num_radiuses_circles * num_sources_in_circle, harmonies.shape[
        0]))  # Note! the order is: inner loop -> outer loop -> inner loop -> outer loop -> ...
    num_rotors = int(magnitudes_in.shape[1] / num_sources_in_circle)
    signals = multibatch_create_harmonic_signal_matrix_style_with_coeff(omega, magnitudes_in_flattened.T, phi_0=phies0_flattened.T,
                                                             duration=duration, harmonies=harmonies, fs=fs,
                                                             modulate_phase=modulate_phase, num_rotors=num_rotors)
    return signals


# jit to accelarate the computation of signals creaation
create_signals_jitted = jit(create_signals, static_argnums=(0, 3, 4, 5, 6))


def create_signals_and_simulate_recordings(rir, num_mics,
                                           fs, duration, omega, phies0, magnitudes, harmonies,
                                           num_sources_in_circle, radiuses_circles, delay_sources, max_rec_len,
                                           flip_rotation_direction, modulate_phase=False,
                                           recordings_foreach_rotor=False, use_multi_distance=False, use_all_distances=False):
    # In our case flip_rotation_direction=[0], recordings_foreach_rotor=False
    signals = create_signals_torch(omega, phies0, magnitudes, duration,
                             fs, num_sources_in_circle, radiuses_circles.shape[0], harmonies, modulate_phase)

    for i in range(radiuses_circles.shape[0]):
        if flip_rotation_direction[i // 2]:
            # Given that we did not incur in this case, we did not change the code to be compatible with torch
            raise NotImplementedError()
            signals = signals.at[num_sources_in_circle * i:num_sources_in_circle * (i + 1)].set(
                signals[num_sources_in_circle * i:num_sources_in_circle * (i + 1)][::-1])

    if recordings_foreach_rotor:
        # Given that we did not incur in this case, we did not change the code to be compatible with torch
        raise NotImplementedError()
        recordings = []
        for i in range(radiuses_circles.shape[0] // 2):
            cur_rotor_recordings = modified_simulate(
                rir[:, num_sources_in_circle * 2 * i:num_sources_in_circle * (2 * i + 2)],
                signals[num_sources_in_circle * 2 * i:num_sources_in_circle * (2 * i + 2)],
                delay_sources, num_sources_in_circle * 2, num_mics, max_rec_len)

            recordings.append(cur_rotor_recordings)
        recordings = jnp.array(recordings)  # shape: (#rotors, #mics, #sig_len)
    else:
        # recordings = modified_simulate(rir, signals, delay_sources, num_sources_in_circle * radiuses_circles.shape[0],
        #                                num_mics, max_rec_len)
        if not use_multi_distance:
            recordings = torch_mb_modified_simulate_single_microphone(rir, signals, delay_sources, num_sources_in_circle * radiuses_circles.shape[0],
                                       num_mics, max_rec_len)
        else:
            out = torch_mb_modified_simulate_multi_microphones(rir, signals, delay_sources, num_sources_in_circle * radiuses_circles.shape[0],
                                       num_mics, max_rec_len, use_all_distances=use_all_distances)
            if use_all_distances:
                recordings = out[0]
                premix_signals = out[1]
                return recordings, premix_signals
            else:
                recordings = out
    
    return recordings