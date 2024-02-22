import os, sys
import numpy as onp
from itertools import product
import argparse
import pdb

from jax import jit
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

# import local methods
from RealRecordings import RealRecordingsSeveralExperiments
from load_tfevent import TFeventParams
from jax_scipy_minimize_forward import minimize
cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from pyroomacoustics_differential import room, forward_model_2D_interface, consts

# Not original imports
import pprint

def add_args(parser):
    '''
    Adding arguments from the user or by default.
    Note that the arguments can be read from a json file as well.
    '''
    #TODO: add the option of reading args from json.
    # gpu args
    parser.add_argument('-gpu', default='0', type=str)
    # optimization mode 
    parser.add_argument('-optimize', action='store_true')
    parser.add_argument('-exp_name', default="", type=str) # to load OR save as
    
    parser.add_argument('-num_rotors', default=1, type=int)
    parser.add_argument('-e_absorption', default=0, type=float)
    
    parser.add_argument('-saved_data_dir_path', default='tmp_single_circle', type=str)
    parser.add_argument('-save_simulated_data', default='', type=str) # for indoor only
    parser.add_argument('-indoor_mode', default='indoor', type=str) # for indoor simulating only
    parser.add_argument('-second_rad', default=0.1, type=float)
    parser.add_argument('-mics_rads', default=[], nargs='+', type=float) # original dists: [0.53 0.57 0.63 0.68 0.73 0.83 0.93 1.03]
    parser.add_argument('-grid_mics_y', default=-1, type=float)
    parser.add_argument('-delta_mics', default=0.02, type=float)
    parser.add_argument('-mics_around_rotor_num', default=0, type=int)
    parser.add_argument('-phase_shift', default=[0, 0], nargs='+', type=float) # degree - phase shift for the second rad
    parser.add_argument('-flip_rotation_direction', default=[0], nargs='+', type=int) # rotation direction for each simulated rotor
    parser.add_argument('-modulate_phase', default=None, nargs='+', type=int)

    # simulated sined or real recordings - choose
    # rotor/ sources args
    parser.add_argument('-rotor_type', default=18, type=int)
    parser.add_argument('-recordings_mode', default='CCW', type=str)
    parser.add_argument('-num_sources_power', default=4, type=int)
    # emitted signals args
    parser.add_argument('-channel', default=-1, type=int)
    parser.add_argument('-vel_rate', default=5, type=int)
    parser.add_argument('-duration', default=0.6, type=float)
    parser.add_argument('-low_cut_coeff', default= -1, type=float)
    parser.add_argument('-high_cut_coeff', default= 3.25, type=float)
    parser.add_argument('-filter_order', default=1, type=int)
    # emitted signals OPTIMIZED params args
    parser.add_argument('-num_phases_mics', default=4, type=int)
    parser.add_argument('-opt_harmonies', default=[], nargs='+', type=float)
    parser.add_argument('-opt_harmonies_init_vals', default=[], nargs='+', type=float)
    parser.add_argument('-opt_phi_0_init_vals', default=[], nargs='+', type=float)
    parser.add_argument('-given_radiuses', default=[-1], nargs='+', type=float)
    # room args
    parser.add_argument('-room_x', default=5., type=float)
    parser.add_argument('-room_y', default=5., type=float)
    parser.add_argument('-org_x', default=2.5, type=float)
    parser.add_argument('-org_y', default=2.5, type=float)
    parser.add_argument('-max_order', default=0, type=int)

    # optimization args
    parser.add_argument('-opt_method', default='L-BFGS-B', type=str)
    parser.add_argument('-step_log_signals', default=2, type=int)
    
    args = parser.parse_args()
    return vars(args) # vars converts the return object from parser.parse_args() to a dict


def create_opt_params(radiuses_circles, phi_init_0, magnitudes):
    '''
    Create opt params in the correct format.
    JAX version.
    '''
    return jnp.concatenate([phi_init_0, magnitudes]), radiuses_circles

def create_opt_params_onp(radiuses_circles, phi_init_0, magnitudes):
    '''
    Create opt params in the correct format.
    NumPy version.
    '''
    return onp.concatenate([phi_init_0, magnitudes]), radiuses_circles

def get_opt_params(opt_params, idx_block, given_radiuses_circles=None):
    '''
    Get opt params by the correct foramt.
    '''
    return given_radiuses_circles, opt_params[:idx_block], opt_params[idx_block:]

def read_optimized_params(exp_name, omega, fs, signals_duration, num_sources_power, radiuses_circles, opt_harmonies):
    tf_event_params = TFeventParams(exp_name=exp_name, omega=omega, fs=fs, duration=signals_duration,
                                    num_sources_power=num_sources_power,
                                    opt_harmonies=opt_harmonies, given_radiuses_circles=radiuses_circles)

    _, radiuses_circles, optimized_phi_0, optimized_harmonies_coeffs, _ = tf_event_params.get_params()
    opt_params, radiuses_circles = create_opt_params(radiuses_circles, optimized_phi_0, optimized_harmonies_coeffs)
    return opt_params, radiuses_circles

def read_optimized_params_onp(exp_name, omega, fs, signals_duration, num_sources_power, radiuses_circles, opt_harmonies):
    tf_event_params = TFeventParams(exp_name=exp_name, omega=omega, fs=fs, duration=signals_duration,
                                    num_sources_power=num_sources_power,
                                    opt_harmonies=opt_harmonies, given_radiuses_circles=radiuses_circles)

    _, radiuses_circles, optimized_phi_0, optimized_harmonies_coeffs, _ = tf_event_params.get_params()
    opt_params, radiuses_circles = create_opt_params_onp(radiuses_circles, optimized_phi_0, optimized_harmonies_coeffs)
    return opt_params, radiuses_circles

@jit
def compute_loss(real_recordings, simulated_recordings):
    '''
    Calculate L2 error in the time domain.
    This function is jitted to accelerate computation.
    '''
    min_size = min([real_recordings.shape[-1], simulated_recordings.shape[-1]])
    real_recordings_cut = real_recordings[:,:min_size]
    simulated_recordings_cut = simulated_recordings[:,:min_size]
    l2_term = jnp.linalg.norm(jnp.subtract(real_recordings_cut, simulated_recordings_cut))
    return l2_term

def forward_func(opt_params, rir, delay_sources, num_mics, max_rec_len,
                    omega, phies0, real_recordings,
                    num_sources, fs, duration, opt_harmonies=[1], phase_shift=[0], flip_rotation_direction=[0],
                    given_radiuses_circles=None, compare_real=True, 
                    return_sim_signals=False, num_rotors=1, modulate_phase=False, recordings_foreach_rotor=False):
    '''
    Computing the forward model - JAX version.
    In this function the optimization params are set to the correct format,
    the signals are created and simulated (convolved) with the given RIR,
    and the loss is computed.
    Return value is: 
    - the simulated recordings by the mics (setting compare_real to False)
    - the loss (setting return_sim_signals to True)
    - both (setting compare_real and return_sim_signals to False)
    '''
    radiuses_circles, phi_init_0, magnitudes = get_opt_params(opt_params, num_sources, given_radiuses_circles)
    if num_rotors > 1:
        phies0 = (jnp.tile(phies0, (num_rotors,1)) + jnp.expand_dims(phase_shift, 1)).reshape(num_sources*num_rotors)
        phi_init_0 = jnp.tile(phi_init_0, (num_rotors, 1, 1))
        magnitudes = jnp.tile(magnitudes, (num_rotors,1,1))
    else:
        phies0 += phase_shift[0]
    phies0 = phies0.reshape(phies0.shape[0], 1, 1) + phi_init_0
    simulated_recordings = forward_model_2D_interface.create_signals_and_simulate_recordings(rir, num_mics, 
                                            fs, duration, omega, phies0, magnitudes, opt_harmonies, 
                                            num_sources_in_circle=num_sources, 
                                            radiuses_circles=radiuses_circles, delay_sources=delay_sources, flip_rotation_direction=flip_rotation_direction,
                                            max_rec_len=max_rec_len, modulate_phase=modulate_phase, recordings_foreach_rotor=recordings_foreach_rotor)
    
    if not compare_real:
        return simulated_recordings
        
    loss = compute_loss(real_recordings, simulated_recordings)

    if return_sim_signals:
        return loss, simulated_recordings
    
    return loss

def build_room_and_read_gt_data(args):
    '''
    Get GT data (real recordings database),
    build the room geometry given the args,
    and set rotor (sources) params.
    NumPy version.
    '''
    exp_name = args['exp_name']
    print(f'Reading experimental dataset from 13.09.2021')
    # QUESTION: what is "radiuses"?
    radiuses = [str(x) for x in sorted(list(range(70,125,10)) + [75, 85])]
    data_dir_names = ['13_9_'+str(x) for x in radiuses]
    data_dir_paths = [os.path.join(cur_dir,'..','..','data','real_recordings', x) for x in data_dir_names]

    recordings_dataset = RealRecordingsSeveralExperiments(data_dir_paths, 
                                                    channel=args['channel'],
                                                    # default in args: -1 (it tries to apply a low-pass filter)
                                                    low_cut_coeff=args['low_cut_coeff'], 
                                                    high_cut_coeff=args['high_cut_coeff'],
                                                    # default in args: 18 (used to filter data with the relevant motor type)
                                                    rotor_type=args['rotor_type'],
                                                    # default in args: 'CCW'(used to filter data with the relevant rotation direction
                                                    # and to do other things)
                                                    recordings_mode=args['recordings_mode'],
                                                    # default in args: 5 
                                                    vel_rate=args['vel_rate'],
                                                    filter_order=args['filter_order'],
                                                    num_phases = args['num_phases_mics'])
    real_recordings, encoder_readings, fs, mics_distances, phi_mics = recordings_dataset.get_avg_audio(args['duration'])
    omega = recordings_dataset.bpf

    # room params
    room_x = args['room_x']
    room_y = args['room_y']
    corners = onp.array([[0,0], [0, room_y], [room_x, room_y], [room_x,0]]).T  # [x,y]

    room2D = room.from_corners(corners, fs=fs, e_absorption=args['e_absorption']) # create room

    # rotor params
    rotor_length_meter = consts.inch_to_meter(args['rotor_type'])
    num_sources_power = args['num_sources_power']
    
    num_rotors = args['num_rotors']
    if num_rotors == 1: # single rotor
        origin2D = onp.array([[args['org_x'], args['org_y']], [args['org_x'], args['org_y']]])
    elif num_rotors == 2: # two rotors
        delta_rad = 1.5 * rotor_length_meter/2 # dist between rotors
        #delta_rad = args['second_rad'] + 0.125
        origin2D = onp.array(   [[args['org_x'] - delta_rad, args['org_y']], [args['org_x'] - delta_rad, args['org_y']],
                                [args['org_x'] + delta_rad, args['org_y']], [args['org_x'] + delta_rad, args['org_y']]])
    elif num_rotors == 4: # four rotors
        delta_rad = 1.5 * rotor_length_meter/2 # dist between rotors
        origin2D = onp.array([  [args['org_x'] - delta_rad, args['org_y'] + delta_rad], [args['org_x'] - delta_rad, args['org_y'] + delta_rad],
                                [args['org_x'] - delta_rad, args['org_y'] - delta_rad], [args['org_x'] - delta_rad, args['org_y'] - delta_rad],
                                [args['org_x'] + delta_rad, args['org_y'] + delta_rad], [args['org_x'] + delta_rad, args['org_y'] + delta_rad],
                                [args['org_x'] + delta_rad, args['org_y'] - delta_rad], [args['org_x'] + delta_rad, args['org_y'] - delta_rad]])
        # rotors order by position:
        # 0 2
        # 1 3
    
    else: # not supported
        raise ValueError(f'{num_rotors} rotors are not supported. Use only 1,2,4 rotors.')
    
    phase_shift = onp.deg2rad(args['phase_shift'])
    num_sources = 2**num_sources_power

    if args['given_radiuses'] == [-1]:
        radiuses_circles = onp.tile(onp.array([rotor_length_meter/2, args['second_rad']]), num_rotors) #TODO ask Tom
    else:
        radiuses_circles = onp.array(args['given_radiuses'])

    # signals params
    signals_duration = args['duration'] # seconds

    opt_harmonies = args['opt_harmonies']
    step_1 = 1/(2**(num_sources_power-2)) # for 2^(5+2)=128 sources in the array #TODO ask Tom
    phies0 = onp.arange(-2 + step_1, 2+step_1, step_1) * onp.pi
    
    delays = onp.zeros(num_sources*radiuses_circles.shape[0])

    return num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
                        fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name

def optimization_process(opt_harmonies_init_vals, opt_phi_0_init_vals, 
                        num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, 
                        delays, mics_distances, phi_mics,encoder_readings,
                        fs, signals_duration, omega, phies0, phase_shift, 
                        real_recordings, exp_name, opt_method, step_log_signals):
    '''
    The optimization loop.
    '''
    # init opt params
    magnitudes = jnp.zeros((num_sources, radiuses_circles.shape[0], (len(opt_harmonies))))
    magnitudes = magnitudes.at[:,].set(opt_harmonies_init_vals)
    phi_0_init_vals = jnp.zeros_like(magnitudes)
    phi_0_init_vals = phi_0_init_vals.at[:,].set(opt_phi_0_init_vals)
    
    opt_params, radiuses_circles = create_opt_params(radiuses_circles, phi_0_init_vals, magnitudes)
    
    # create rir once, since the positions of sources and mics aren't changed
    _, mics_array, rir, _ = forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(origin2D, origin2D,
                                        room2D, delays,
                                        mics_distances, phi_mics=phi_mics, 
                                        coords_mics=None, num_mics_circ_array=0, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
                                        sources_distances=None, phi_sources_array=None, phi0_sources=0,
                                        fs=fs, max_order=0,
                                        enable_prints=True, is_plot_room=True, room_plot_out='mics_plotted.png')
                                        

    print('Starting optimization process')
    # compute borders signals for Tensorboard logging - four spins of the rotor
    home_pos_rotor = jnp.where(encoder_readings == 0)[1]
    start_spin_id = 3 #TODO - ask Tom
    num_spins = 5
    
    num_mics = mics_array['M']
    max_len_rir = max([len(rir[i][j])
                        for i, j in product(range(num_mics), range(len(rir[0])))])
    
    # pad rir with zeros to make a numpy - will be used later on for convolving
    rir = jnp.array([[jnp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir[j]] for j in range(len(rir))])

    max_signal_len = int(fs * signals_duration)
    max_rec_len = max_len_rir + max_signal_len
    if max_rec_len % 2 == 1:
        max_rec_len += 1

    delays_rec = 0 #NOTE: not supporting multiple delays for microphones (no need to in our case)

    borders_signals = (home_pos_rotor[start_spin_id], home_pos_rotor[start_spin_id+num_spins-1])
    minimize(forward_func, opt_params, method=opt_method,options={'ftol': 1e-5, 'gtol': 1e-5, 'disp': False},
                    args=(rir, delays_rec, num_mics, int(max_rec_len),
                            omega, phies0, real_recordings,
                            num_sources, fs, signals_duration, jnp.asarray(opt_harmonies), [0], [0],
                            radiuses_circles, True),
                            exp_name=exp_name,
                            opt_harmonies_list=opt_harmonies,
                            idx_block_opt_params=num_sources,
                            step_log_signals=step_log_signals,
                            borders_signals=borders_signals,
                            get_opt_params=get_opt_params)
    print('Done optimization')

def run_forward_optimized_params(opt_params, mics_distances, phi_mics, real_recordings, encoder_readings, origin2D, room2D, delays,radiuses_circles, num_sources,
                                fs, signals_duration, omega, phies0, opt_harmonies, phase_shift, grid_mics_y, delta_mics, num_phases_mics, mics_rads, exp_name, num_rotors=1, flip_rotation_direction=[0],
                                max_order=0, modulate_phase=False):
    '''
    Run forward model with optimized params.
    '''
    #TODO: define mics distances as a parameter
    if len(mics_rads) > 0:
        mics_distances = jnp.array(mics_rads)
        #mics_distances = jnp.array([1.03, 1.2, 1.4, 1.6, 1.8, 2.])
        phi_mics = jnp.tile(phi_mics[:num_phases_mics], len(mics_distances))
        mics_distances = jnp.repeat(mics_distances, num_phases_mics)

    # saving real data
    print('Saving ground truth data in an easy to use form')
    data_dir_path = os.path.join(cur_dir, 'optimized_data') #TODO:
    real_path = os.path.join(data_dir_path, 'real_sines.npy')
    jnp.save(real_path, real_recordings)
    encoder_path = os.path.join(data_dir_path, 'encoder_readings.npy')
    jnp.save(encoder_path, encoder_readings)

    # create rir once, since the positions of sources and mics aren't changed
    _, mics_array, rir, _ = forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(
                                        origin2D, origin2D,
                                        room2D, delays,
                                        mics_distances, phi_mics=phi_mics, 
                                        coords_mics=None, num_mics_circ_array=0, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
                                        sources_distances=None, phi_sources_array=None, phi0_sources=0,
                                        fs=fs, max_order=max_order,
                                        grid_mics_y=grid_mics_y, delta_mics=delta_mics, mics_R=max(radiuses_circles), 
                                        enable_prints=True, is_plot_room=True, room_plot_out='mics_plotted.png')

    num_mics = mics_array['M']
    max_len_rir = max([len(rir[i][j])
                        for i, j in product(range(num_mics), range(len(rir[0])))])
    
    # pad rir with zeros to make a numpy - will be used late on for convolving
    rir = jnp.array([[jnp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir[j]] for j in range(len(rir))])

    max_signal_len = int(fs * signals_duration)
    max_rec_len = max_len_rir + max_signal_len
    if max_rec_len % 2 == 1:
        max_rec_len += 1

    delays_rec = 0 #delays_rec = int(jnp.floor(delays * fs))
    
    print(f'Simulating recordings by exp: {exp_name}')
    # simulate recordings based on fitted params
    simulated_recordings = forward_func(opt_params, rir, delays_rec, num_mics, int(max_rec_len),
                            omega, phies0, real_recordings,
                            num_sources, fs, signals_duration, jnp.asarray(opt_harmonies), phase_shift, flip_rotation_direction,
                            radiuses_circles, compare_real=False, num_rotors=num_rotors, modulate_phase=modulate_phase)

    return simulated_recordings


def main_forward(args):
    '''
    The main function.
    '''
    num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, _,\
                        fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)

    if args['optimize']:
        # It added args['flip_rotation_direction'] to the list of arguments but it's not supported!
        # optimization_process(args['opt_harmonies_init_vals'], args['opt_phi_0_init_vals'], num_sources, radiuses_circles, opt_harmonies, origin2D, room2D,
        #                     delays, mics_distances, phi_mics, encoder_readings, fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name,
        #                     args['opt_method'], args['step_log_signals'], args['flip_rotation_direction'])
        optimization_process(args['opt_harmonies_init_vals'], args['opt_phi_0_init_vals'], num_sources, radiuses_circles, opt_harmonies, origin2D, room2D,
                            delays, mics_distances, phi_mics, encoder_readings, fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name,
                            args['opt_method'], args['step_log_signals'])

    else: # load previously optimized params - logged by TBoard
        # read params from tb
        opt_params, radiuses_circles = read_optimized_params(exp_name, omega, fs, signals_duration, args['num_sources_power'], radiuses_circles, opt_harmonies)
        simulated_recordings = run_forward_optimized_params(opt_params, mics_distances, phi_mics, real_recordings, encoder_readings, 
                                                            origin2D, room2D, delays,radiuses_circles, num_sources,
                                                            fs, signals_duration, omega, phies0, opt_harmonies, phase_shift, 
                                                            args['grid_mics_y'], args['delta_mics'], args['num_phases_mics'], args['mics_rads'], 
                                                            exp_name, args['num_rotors'], args['flip_rotation_direction'], max_order=args['max_order'], modulate_phase=args['modulate_phase'])
        simulated_rec_dir = os.path.join(os.path.join(cur_dir, 'optimized_data'), 'simulated_rec.npy')
        jnp.save(simulated_rec_dir, simulated_recordings)
        return simulated_recordings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu'] #'2, 3'#
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    main_forward(args)

# optimize:
# python forward_model.py -gpu 0 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -channel 0 -duration 0.5 -step_log_signals 1 -num_phases_mics 64 -optimize 
# no optimization:
# python forward_model.py -gpu 0 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -channel 0 -duration 0.5 -num_phases_mics 8
