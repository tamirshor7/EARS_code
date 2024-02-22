import EARS.forward_model.forward_model as forward_model
import argparse
import os
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
from multiprocessing import Pool
import EARS.pyroomacoustics_differential.forward_model_2D_interface as forward_model_2D_interface
from itertools import product
import concurrent.futures
import torch
from tqdm import tqdm


def _create_harmonic_signal_matrix_style_with_coeff(omega, coeffs, phi_0=0, harmonies=[1], fs=44100, duration=0.5):
    signals_len = int(duration * fs)
    t = jnp.expand_dims(jnp.linspace(0, duration, signals_len), 0)
    
    harmonies = jnp.array(harmonies)
    
    harmonies = harmonies[:, jnp.newaxis]
    
    harmonies_omega_t = jnp.tile(harmonies * 2 * jnp.pi * omega * t, (phi_0.shape[1],1,1))
    
    cos_res = jnp.cos(harmonies_omega_t + jnp.expand_dims(phi_0.T,2))
    
    samples = jnp.squeeze((jnp.expand_dims(coeffs.T,1) @ cos_res), 1)
    return samples

def _create_signals(omega, phies0, magnitudes_in, duration=0.5, fs=44100,  num_sources_in_circle=4, num_radiuses_circles=1, harmonies=[1],flip_rotation_direction=None):
    phies0_flattened = jnp.reshape(phies0, (num_radiuses_circles * num_sources_in_circle, harmonies.shape[0]))
    magnitudes_in_flattened = jnp.reshape(magnitudes_in, (num_radiuses_circles * num_sources_in_circle, harmonies.shape[0]))
    signals = _create_harmonic_signal_matrix_style_with_coeff(omega, magnitudes_in_flattened.T, phi_0=phies0_flattened.T, duration=duration, harmonies=harmonies, fs=fs)
    if flip_rotation_direction is not None:
        for i in range(num_radiuses_circles):
            if flip_rotation_direction[i//2]:
                signals = signals.at[num_sources_in_circle * i:num_sources_in_circle * (i+1)].set(signals[num_sources_in_circle * i:num_sources_in_circle * (i+1)][::-1])
    return signals

def _shear(xxyy, angle):
    '''
    Shear a set of points by angle in degrees around the origin
    :param xxyy: set of points stored in shape (num_points, 2)
    :param angle: the rotating angle in degrees
    :return rotated points
    '''
    angle = np.deg2rad(angle)
    c = 1/np.tan(angle)
    R = np.array([
        [1, c],
        [0, 1],
        ])  
    return (R@(xxyy.T)).T

def _get_specific_params(x:float, y:float, origin2D) -> tuple:
    origin_sources: np.ndarray = origin2D + np.array([[x, y]])
    origin_mics: list = [np.array([x,y])]
    return origin_sources, origin_mics

def _compute_ISM_and_RIR_phased_sources_array(origin_sources: np.ndarray, origin_mics: list, 
        angle:float, delays, num_sources, num_mics, mics_R, radiuses_circles, fs, room2D,
        max_order:int = 1) -> tuple:
    sources = forward_model_2D_interface.add_sources_phased_rotor_array_no_signals(origin_sources,
                            delays, num_sources_in_circle=num_sources, radiuses_circles=radiuses_circles,
                            distances=None, phi_array=None, phi0_sources=angle,
                            enable_prints=False)
    
    mics_array = forward_model_2D_interface.add_mics_to_room(origin_mics, fs=fs, distances=None, phi_mics = angle, coords=None, mics_R=mics_R,
                                enable_prints=False, num_mics_circ_array=num_mics, 
                                grid_mics_y=-1, delta_mics=2, room_max_x=room2D['corners'][0][0],
                                mics_around_rotor_num=0)

    visibility, sources = forward_model_2D_interface.image_source_model_onp(room2D, sources, mics_array, max_order=1)
    
    rir, _ = forward_model_2D_interface.compute_rir_threaded(sources, mics_array, visibility, room_fs=room2D['fs'], room_t0=room2D['t0'], max_order=max_order)

    return rir

def get_rir(x:float, y:float, angle:float, origin2D, delays, mics_R, radiuses_circles, fs, room2D,
            num_mics:int=8, max_order:int=1) -> np.ndarray:
    origin_sources, origin_mics = _get_specific_params(x,y, origin2D)
    rir = _compute_ISM_and_RIR_phased_sources_array(origin_sources, origin_mics, 
                                           delays=delays, num_mics=num_mics, max_order=max_order,
                                           mics_R=mics_R, radiuses_circles=radiuses_circles, fs=fs, room2D=room2D,
                                           angle=angle)
    max_len_rir = max([len(rir[i][j])
                        for i, j in product(range(num_mics), range(len(rir[0])))])
    rir = np.array([[np.pad(i, (0, int(max_len_rir)-len(i))) for i in rir[j]] for j in range(len(rir))])
    return rir

# @jit
# def convolve(rir_at_loc, signals):
#     output_signals = jnp.reshape(signals, (1,signals.shape[0], signals.shape[1]))
#     rir_at_loc_flipped = jnp.flip(rir_at_loc, -1)
#     premix_signals = lax.conv_general_dilated(output_signals, rir_at_loc_flipped, (1,), 'VALID')
#     output_signals = jnp.sum(premix_signals, axis=0)
#     return output_signals

def convolve(rir_at_loc, signals):
    output_signals = torch.tensor(np.copy(signals)).unsqueeze(0)
    rir = torch.tensor(np.copy(rir_at_loc))
    rir_at_loc_flipped = torch.flip(rir, [-1])
    premix_signals = torch.nn.functional.conv1d(output_signals, rir_at_loc_flipped, padding=0)
    output_signals = torch.sum(premix_signals, dim=0)
    output_signals = output_signals.detach().cpu().numpy()
    return output_signals

def process_location_and_angle(data_in):
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    ## Unpack data
    xy, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, save_dir, SIGNALS = data_in
    x = xy[0]
    y = xy[1]
    ##

    ## Create 12 random angles and 0,90,180,270
    number_of_random_angles: int = 12
    angles: np.ndarray = np.random.default_rng().uniform(0,2*np.pi,number_of_random_angles)
    angles = np.sort(np.concatenate((angles, np.array([0, np.pi/2, np.pi, 3*np.pi/2]))))
    ##

    ## Compute other parameters
    origin_mics_str: str = f'{int(x*10**8)}_{int(y*10**8)}'

    for angle in angles:
        ## Compute RIRs
        origin_sources, origin_mics = _get_specific_params(x,y, origin2D)
        rir = _compute_ISM_and_RIR_phased_sources_array(origin_sources, origin_mics,
                                                        delays=delays, num_sources=num_sources, num_mics=num_mics, max_order=1,
                                                        mics_R=mics_R, radiuses_circles=radiuses_circles, fs=fs, room2D=room2D,
                                                        angle=angle)
        max_len_rir = max([len(rir[i][j])
                        for i, j in product(range(num_mics), range(len(rir[0])))])
        rir = np.array([[np.pad(i, (0, int(max_len_rir)-len(i))) for i in rir[j]] for j in range(len(rir))])
        ##
        ## Convolve with signals
        output_signals = convolve(rir, SIGNALS)
        ##
        ## Save
        np.save(os.path.join(save_dir, f'{origin_mics_str}_{int(angle*10**8)}.npy'), output_signals)
        ##

def compute_pressure_foreach_location_and_angle():
    ### Compute signals so that it's accessible globally
    print('Building the signals')
    
    ## Constants
    OMEGA: float = 46.9333688083713
    PHIES0: np.ndarray = np.load('/mnt/walkure_public/tamirs/EARS_assets/phies0.npy')
    MAGNITUDES: np.ndarray = np.load('/mnt/walkure_public/tamirs/EARS_assets/magnitudes.npy')
    SIGNALS_DURATION: float = 0.5
    FS:float = 3003.735603735763
    NUM_SOURCES:int = 128
    RADIUSES_CIRCLES: np.ndarray = np.array([0.2286, 0.51  , 0.2286, 0.51  , 0.2286, 0.51  , 0.2286, 0.51  ])
    OPT_HARMONIES: jnp.ndarray = jnp.asarray([0.5, 1.0, 2.0, 3.0])
    FLIP_ROTATION_DIRECTION: list = [1, 0, 0, 1]
    ##

    SIGNALS:jnp.ndarray = _create_signals(OMEGA, PHIES0, MAGNITUDES, SIGNALS_DURATION, 
                            FS, NUM_SOURCES, RADIUSES_CIRCLES.shape[0], OPT_HARMONIES, flip_rotation_direction=FLIP_ROTATION_DIRECTION)
    ###

    ### Compute all of the necessary locations
    print('Building the room')
    num_mics = args['num_phases_mics']
    margins = args['margins']
    num_points_per_side = args['num_points_per_side']
    delta = args['delta']
    rotating_angle = args['rotating_angle']

    num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = forward_model.build_room_and_read_gt_data(args)
    mics_R = 2*rotor_length_meter
    origin2D -= np.array([[args['org_x'], args['org_y']]])
    if num_points_per_side is None:
        xx = np.arange( round((mics_R+margins), 2), round((args['room_x']-mics_R-margins),2), delta)
        yy = np.arange( round((mics_R+margins), 2), round((args['room_y']-mics_R-margins),2), delta)
    else:
        if rotating_angle is not None:
            cotan_theta = 1/np.tan(np.deg2rad(rotating_angle))

            y_min = round((mics_R+margins), 2)
            x_min = round(cotan_theta*(y_min + 1/cotan_theta*args['room_x']-np.sqrt(1+1/cotan_theta**2)*(mics_R+margins)) - cotan_theta*y_min, 2)

            y_max = round((args['room_y']-mics_R-margins),2)
            x_max = round(cotan_theta*(y_min + np.sqrt(1+1/cotan_theta**2)*(mics_R+margins)) - cotan_theta*y_min, 2)
            
            xx = np.linspace(x_min, x_max, num_points_per_side)
            yy = np.linspace(y_min, y_max, num_points_per_side)
        else:
            xx = np.linspace( round((mics_R+margins), 2), round((args['room_x']-mics_R-margins),2), num_points_per_side)
            yy = np.linspace( round((mics_R+margins), 2), round((args['room_y']-mics_R-margins),2), num_points_per_side)
    xxyy = np.array(np.meshgrid(xx,yy)).T.reshape(-1,2)

    if rotating_angle is not None:
        xxyy = _shear(xxyy, rotating_angle)
        xxyy = np.round(xxyy,2)


    save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    os.makedirs(save_dir, exist_ok=True)

    process_x = lambda x: (int(x.strip().split('.')[0].split('_')[0]), int(x.strip().split('.')[0].split('_')[1]))
    already_processed = [process_x(x) for x in sorted(os.listdir(save_dir))]
    xxyy = [x for x in xxyy if (int(x[0]*10**8), int(x[1]*10**8)) not in already_processed]
    ###

    ### Parallelize process_location_and_angle
    data_inputs = [[xy, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, save_dir, SIGNALS] for xy in xxyy]
    print(f"There are {len(data_inputs)} locations to compute")
    # cpu_pool = int(os.cpu_count()/4)
    # with Pool(cpu_pool) as pool:
    #     pool.map(process_location_and_angle, data_inputs)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #         executor.map(process_location_and_angle, data_inputs)
    for data_input in tqdm(data_inputs):
        process_location_and_angle(data_input)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = forward_model.add_args(parser)
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    
    compute_pressure_foreach_location_and_angle()