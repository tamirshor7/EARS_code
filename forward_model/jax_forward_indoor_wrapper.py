import numpy as onp
import jax.numpy as jnp
import os, sys
import argparse
from itertools import product
from multiprocessing import pool
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
import pdb
import time
# import local methods
from forward_model import add_args, build_room_and_read_gt_data, read_optimized_params_onp, forward_func
cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from pyroomacoustics_differential import jax_forward_model_2D_interface

# RIR:
# python EARS/forward_model/forward_indoor_wrapper.py -gpu 0 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 2 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels -phase_shift 0 0 -e_absorption 0.2
# RIR BY DEMAND:
# python forward_indoor_wrapper.py -max_order 20 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 1 -num_phases_mics 8 -saved_data_dir_path rir_indoor -phase_shift 0 0 -e_absorption 0.2
# SIMULATION
# Tom's simulation
# python forward_indoor_wrapper.py -gpu 3 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 4 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels_4_rotors -phase_shift 0 0 0 0 -save_simulated_data grid_db_same_phase_cut -indoor_mode indoor -e_absorption 0.2 -flip_rotation_direction 0 1 1 0 -modulate_phase
# Our simulation
# python forward_indoor_wrapper.py -gpu 3 -max_order 7 -exp_name Jan10_11-26-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 4 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels_4_rotors -phase_shift 0 0 0 0 -save_simulated_data grid_db_same_phase_cut -indoor_mode indoor -e_absorption 0.2 -flip_rotation_direction 0 1 1 0 -modulate_phase

# python forward_indoor_wrapper.py -gpu 3 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 2 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels -phase_shift 0 0 -save_simulated_data grid_db_same_phase_cut -indoor_mode indoor -e_absorption 0.2 -flip_rotation_direction 0 1

# python forward_indoor_wrapper.py -gpu 0 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 2 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels -phase_shift 0 0 -modulate_phase -save_simulated_data grid_db_phase_modulation_cut -indoor_mode indoor -e_absorption 0.2
# python forward_indoor_wrapper.py -gpu 1 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 2 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels -phase_shift 0 45 -save_simulated_data grid_db_phase_0_45_cut -indoor_mode indoor -e_absorption 0.2
# python forward_indoor_wrapper.py -gpu 1 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 2 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels -phase_shift -45 45 -save_simulated_data grid_db_phase_minus45_45_cut -indoor_mode indoor -e_absorption 0.2
# python forward_indoor_wrapper.py -gpu 1 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 2 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels -phase_shift 0 -45 -save_simulated_data grid_db_phase_0_minus45_cut -indoor_mode indoor -e_absorption 0.2

# SIMULATION BY DEMAND:
# python forward_indoor_wrapper.py -gpu 3 -max_order 15 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 1 -num_phases_mics 8 -saved_data_dir_path rir_indoor15 -phase_shift 0 0 -e_absorption 0.2
def process_location(data_in):
    xy, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, rir_dirs = data_in
    x = xy[0]
    y = xy[1]
    origin_sources = origin2D + jnp.array([[x, y]])
    origin_mics = [jnp.array([x,y])]
    # don't use round here for x,y! this will raise an error in the positions collection.
    print(f'Processing position: {origin_mics[0]}', flush=True)
    origin_mics_str = f'{int(x*100)}_{int(y*100)}'
    start_time = time.time()
    # UNCOMMENT TO GET RIR BY ORDERS
    # _, _, rir, rir_by_orders = jax_forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(
    #                                     origin_sources, origin_mics,
    #                                     room2D, delays,
    #                                     mics_distances=None, phi_mics=0, 
    #                                     coords_mics=None, num_mics_circ_array=num_mics, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
    #                                     sources_distances=None, phi_sources_array=None, phi0_sources=0, mics_R=mics_R, 
    #                                     fs=fs, max_order=args['max_order'],
    #                                     enable_prints=False, is_plot_room=True, room_plot_out='localization_plotted_room.pdf')
    DEBUG = True
    _, _, rir, _ = jax_forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(
                                        origin_sources, origin_mics,
                                        room2D, delays,
                                        mics_distances=None, phi_mics=0, 
                                        coords_mics=None, num_mics_circ_array=num_mics, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
                                        sources_distances=None, phi_sources_array=None, phi0_sources=0, mics_R=mics_R, 
                                        fs=fs, max_order=args['max_order'],
                                        enable_prints=DEBUG, is_plot_room=True, room_plot_out='localization_plotted_room.pdf')
    print(f'Time: {time.time()-start_time}')
    # by now I expect rir to be a jax array!
    max_len_rir = max([len(rir[i][j])
                        for i, j in product(range(num_mics), range(len(rir[0])))])
    

    rir = onp.as_array([[jnp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir[j]] for j in range(len(rir))])
    breakpoint()
    onp.save(os.path.join(rir_dirs[0], origin_mics_str+'.npy'), rir)

    # UNCOMMENT TO SAVE RIR BY ORDER
    # for k in range(args['max_order']+1):
    #     rir_by_order = onp.as_array([[jnp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir_by_orders[k][j]] for j in range(len(rir_by_orders[k]))])
    #     onp.save(os.path.join(rir_dirs[k+1], origin_mics_str+'.npy'), rir_by_order)
    

def compute_rir_foreach_location(args, delta=0.02, margins=0.02, cpu_pool=40, by_demand=False):
    print('Building the room')
    num_mics = args['num_phases_mics']
    num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)
    
    mics_R = 2*rotor_length_meter
    origin2D -= jnp.array([[args['org_x'], args['org_y']]])
    
    # xx = onp.arange(mics_R+delta, args['room_x']-mics_R-delta, delta)
    # yy = onp.arange(mics_R+delta, args['room_y']-mics_R-delta, delta)
    xx = jnp.arange( round((mics_R+margins), 2), round((args['room_x']-mics_R-margins),2), delta)
    yy = jnp.arange( round((mics_R+margins), 2), round((args['room_y']-mics_R-margins),2), delta)
    

    # create an empty df to save the processed data
    #save_dir = os.path.join('/mnt', 'ssd','tomh', 'rir_data', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #save_dir = os.path.join('/mnt', 'walkure_public','tomh', 'rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    save_dir = os.path.join(cur_dir,'..','..','data','jax_rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    os.makedirs(save_dir, exist_ok=True)
    #save_dir = os.path.join('/mnt', 'walkure_public','tomh', 'rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    rir_indoor_dir = os.path.join(save_dir, 'rir_indoor')
    # UNCOMMENT TO GET RIR BY ORDER
    # rir_dirs = [rir_indoor_dir] + [os.path.join(save_dir, 'rir_images'+str(k)) for k in range(args['max_order']+1)]
    rir_dirs = [rir_indoor_dir]

    for rir_dir_path in rir_dirs:
        if not os.path.exists(rir_dir_path):
            os.makedirs(rir_dir_path)

    xxyy = jnp.array(jnp.meshgrid(xx,yy)).T.reshape(-1,2)
    
    # skip previous-calculaed RIRs
    process_x = lambda x: (int(x.strip().split('.')[0].split('_')[0]), int(x.strip().split('.')[0].split('_')[1]))
    already_processed = [process_x(x) for x in sorted(os.listdir(rir_dirs[-1]))]
    xxyy = [x for x in xxyy if (int(x[0]*100), int(x[1]*100)) not in already_processed]
    # cut xxyy in half
    xxyy = xxyy[int(len(xxyy))//2:]

    if by_demand:
        xxyy = [(2.5,0.93), (0.93,0.93), (2.5,2.5)]

    data_inputs = [[xy, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, rir_dirs] for xy in xxyy]
    print('Computing RIR foreach location in the room (pool is not set!)')
    
    # with ThreadPoolExecutor(cpu_pool) as pool:
    #     pool.map(process_location, data_inputs)
    #print(data_inputs[0])
    #process_location(data_inputs[0])
    for data_in in data_inputs:
        process_location(data_in)

    return num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name

def simulate_signals_with_rir(args, delta=0.02, num_revolutions=4):
    # read params from tb
    num_sources, radiuses_circles, opt_harmonies, _, _, _, _, _, _, _,\
        fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)

    opt_params, radiuses_circles = read_optimized_params_onp(exp_name, omega, fs, signals_duration, args['num_sources_power'], radiuses_circles, opt_harmonies)

    num_mics = args['num_phases_mics']
    num_rotors = args['num_rotors']

    dir_save_path = f'indoor_recordings_{num_rotors}_rotors_{num_mics}_mics_d_{delta}_mod_{args["modulate_phase"]}' #TODO
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)

    #read_dir = os.path.join('/mnt', 'ssd','tomh', 'rir_data', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    read_dir = os.path.join(cur_dir,'..','..','data','rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    rir_dir = os.path.join(read_dir, 'rir_'+args["indoor_mode"])
    s = os.listdir(rir_dir)
    
    rirs_paths = sorted(s, key=lambda single_rir: (int(single_rir.strip().split('.')[0].split('_')[0]), int(single_rir.strip().split('.')[0].split('_')[1])))
    
    # 536 => (133, 253) # paper - random
    # 1984 => (248, 248) # center
    # 0 => (93, 93) # top-left corner
    # 1953 => (248, 93) # center-left
    # 130 => (338, 308) # bottom-right quarter

    print(f'Simulating recordings by exp: {exp_name}')
    for idx, rir_path in enumerate(rirs_paths):
        rir = None
        for order in range(1, args['max_order']+1):
            rir_dir = os.path.join(read_dir, f'rir_images{order}')
            if order == 1:
                rir = onp.load(os.path.join(rir_dir, rir_path))
            else:
                rir += onp.load(os.path.join(rir_dir, rir_path))
        

        origin_x, origin_y = [int(x) for x in rir_path.strip().split('.')[0].split('_')]

        print(origin_x, origin_y)

        max_len_rir = rir.shape[2]
        max_signal_len = int(fs * signals_duration)
        max_rec_len = max_len_rir + max_signal_len
        if max_rec_len % 2 == 1:
            max_rec_len += 1

        delays_rec = 0 #delays_rec = int(jnp.floor(delays * fs))
        
        phase_shift = jnp.array(args['phase_shift'])
        # simulate recordings based on fitted params
        simulated_recordings = forward_func(opt_params, rir, delays_rec, num_mics, int(max_rec_len),
                                omega, phies0, real_recordings,
                                num_sources, fs, signals_duration, jnp.asarray(opt_harmonies), phase_shift, args['flip_rotation_direction'],
                                radiuses_circles, compare_real=False, num_rotors=num_rotors, modulate_phase=args['modulate_phase'], recordings_foreach_rotor=True)
        
        # save simulated data. shape: (#rotors, #mics, #sig_len)
        file_save_path = os.path.join(dir_save_path, f'{origin_x}_{origin_y}.npy')
        onp.save(file_save_path, simulated_recordings)
        

def simulate_by_demand(args):
    # read params from tb
    num_sources, radiuses_circles, opt_harmonies, _, _, _, _, _, _, _,\
        fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)

    opt_params, radiuses_circles = read_optimized_params_onp(exp_name, omega, fs, signals_duration, args['num_sources_power'], radiuses_circles, opt_harmonies)

    num_mics = args['num_phases_mics']
    num_rotors = args['num_rotors']


    read_dir = os.path.join('/mnt', 'ssd','tomh', 'rir_data', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}')
    rir_names = ['rir_indoor']+[f'rir_images{i}' for i in range(args['max_order']+1)]
    save_dir = os.path.join('simulated_signals',f'absorption_{args["e_absorption"]}_maxord{args["max_order"]}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    origins = [(250,250), (93,93), (250,93)]
    

    for origin in origins:
        for rir_name in rir_names:
            rir_path = os.path.join(read_dir, rir_name, f'{origin[0]}_{origin[1]}.npy')
            # if rir_name in rir_names[:2]:
            #     rir = onp.load(rir_path)
            # else:
            #     rir += onp.load(rir_path)
            rir = onp.load(rir_path)
            
            print(origin, rir_name)

            max_len_rir = rir.shape[2]
            max_signal_len = int(fs * signals_duration)
            max_rec_len = max_len_rir + max_signal_len
            if max_rec_len % 2 == 1:
                max_rec_len += 1

            delays_rec = 0 #delays_rec = int(jnp.floor(delays * fs))
            
            phase_shift = jnp.array(args['phase_shift'])
            # simulate recordings based on fitted params
            simulated_recordings = forward_func(opt_params, rir, delays_rec, num_mics, int(max_rec_len),
                                    omega, phies0, real_recordings,
                                    num_sources, fs, signals_duration, jnp.asarray(opt_harmonies), phase_shift, args['flip_rotation_direction'],
                                    radiuses_circles, compare_real=False, num_rotors=num_rotors, modulate_phase=args['modulate_phase'])

            save_path = os.path.join(save_dir, f'{origin[0]}_{origin[1]}_{rir_name}.npy')
            jnp.save(save_path, simulated_recordings)


def correlate_each_channel(x, y):
    res_total = 0
    num_channels = x.shape[0]
    for channel in range(num_channels):
        res, _ = stats.pearsonr(x[channel], y[channel])
        res_total += res
    res_total /= num_channels
    return res_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    # environment options (gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='true'
    compute_rir_foreach_location(args, delta=0.05, cpu_pool=40)
    #simulate_signals_with_rir(args,delta=0.05)
    #simulate_by_demand(args)
