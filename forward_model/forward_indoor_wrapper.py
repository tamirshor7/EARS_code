import numpy as onp
import jax.numpy as jnp
import os, sys
import argparse
from itertools import product
from multiprocessing import Pool
from scipy import stats
import pdb
import time
# import local methods
from forward_model import add_args, build_room_and_read_gt_data, read_optimized_params_onp, forward_func
cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from pyroomacoustics_differential import forward_model_2D_interface
from tqdm import tqdm
from EARS.io import hdf5
import traceback
import sys

from getpass import getuser

from EARS.forward_model.non_convex_room import get_points_in_polygon_from_corners

from EARS.io.fast_io import get_listed_files

#from  scipy.stats.qmc import PoissonDisk
# RIR: (Before num_rotors was set to 2 by Tom, but I think that it's wrong!) (Also before: -saved_data_dir_path rir_indoor_4_channels and -phase_shift 0 0 )
# python forward_indoor_wrapper.py -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 4 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels_4_rotors -phase_shift 0 0 0 0 -e_absorption 0.2
# RIR BY DEMAND:
# python forward_indoor_wrapper.py -max_order 20 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 1 -num_phases_mics 8 -saved_data_dir_path rir_indoor -phase_shift 0 0 -e_absorption 0.2
# SIMULATION
# Tom's simulation
# python forward_indoor_wrapper.py -gpu 3 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 4 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels_4_rotors -phase_shift 0 0 0 0 -save_simulated_data grid_db_same_phase_cut -indoor_mode indoor -e_absorption 0.2 -flip_rotation_direction 0 1 1 0 -modulate_phase 0 0 0 0
# Our simulation
# python forward_indoor_wrapper.py -gpu 3 -max_order 7 -exp_name Jan10_11-26-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 4 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels_4_rotors -phase_shift 0 0 0 0 -save_simulated_data grid_db_same_phase_cut -indoor_mode indoor -e_absorption 0.2 -flip_rotation_direction 0 1 1 0 -modulate_phase 0 0 0 0

# python forward_indoor_wrapper.py -gpu 3 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 2 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels -phase_shift 0 0 -save_simulated_data grid_db_same_phase_cut -indoor_mode indoor -e_absorption 0.2 -flip_rotation_direction 0 1

# python forward_indoor_wrapper.py -gpu 0 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 2 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels -phase_shift 0 0 -modulate_phase -save_simulated_data grid_db_phase_modulation_cut -indoor_mode indoor -e_absorption 0.2
# python forward_indoor_wrapper.py -gpu 1 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 2 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels -phase_shift 0 45 -save_simulated_data grid_db_phase_0_45_cut -indoor_mode indoor -e_absorption 0.2
# python forward_indoor_wrapper.py -gpu 1 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 2 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels -phase_shift -45 45 -save_simulated_data grid_db_phase_minus45_45_cut -indoor_mode indoor -e_absorption 0.2
# python forward_indoor_wrapper.py -gpu 1 -max_order 7 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 2 -num_phases_mics 4 -saved_data_dir_path rir_indoor_4_channels -phase_shift 0 -45 -save_simulated_data grid_db_phase_0_minus45_cut -indoor_mode indoor -e_absorption 0.2

# SIMULATION BY DEMAND:
# python forward_indoor_wrapper.py -gpu 3 -max_order 15 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 1 -num_phases_mics 8 -saved_data_dir_path rir_indoor15 -phase_shift 0 0 -e_absorption 0.2
def compute_direct_component():
    print('Building the room')
    num_mics = args['num_phases_mics']
    num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)
    mics_R = 2*rotor_length_meter
    origin2D -= onp.array([[args['org_x'], args['org_y']]])
    
    # xx = onp.arange(mics_R+delta, args['room_x']-mics_R-delta, delta)
    # yy = onp.arange(mics_R+delta, args['room_y']-mics_R-delta, delta)
    

    # create an empty df to save the processed data
    #save_dir = os.path.join('/mnt', 'ssd','tomh', 'rir_data', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #save_dir = os.path.join('/mnt', 'walkure_public','tomh', 'rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    save_dir = os.path.join(cur_dir,'..','..','data','rir', 'direct_component')
    os.makedirs(save_dir, exist_ok=True)
    #save_dir = os.path.join('/mnt', 'walkure_public','tomh', 'rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    
    # UNCOMMENT TO SAVE RIR BY ORDER (watch also function process_location) 
    # rir_dirs = [rir_indoor_dir] + [os.path.join(save_dir, 'rir_images'+str(k)) for k in range(args['max_order']+1)]
    rir_dirs = [save_dir]

    xxyy = onp.array([[2.48,2.48]])

    data_inputs = [[xy, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, rir_dirs] for xy in xxyy]
    xy, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, rir_dirs = data_inputs[0]
    x = xy[0]
    y = xy[1]
    origin_sources = origin2D + onp.array([[x, y]])
    origin_mics = [onp.array([x,y])]
    # don't use round here for x,y! this will raise an error in the positions collection.
    print(f'Processing position: {origin_mics[0]}')
    start_time = time.time()
    # UNCOMMENT TO SAVE RIR BY ORDERS
    # _, _, rir, rir_by_orders = forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(
    #                                     origin_sources, origin_mics,
    #                                     room2D, delays,
    #                                     mics_distances=None, phi_mics=0, 
    #                                     coords_mics=None, num_mics_circ_array=num_mics, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
    #                                     sources_distances=None, phi_sources_array=None, phi0_sources=0, mics_R=mics_R, 
    #                                     fs=fs, max_order=args['max_order'],
    #                                     enable_prints=False, is_plot_room=True, room_plot_out='localization_plotted_room.pdf')
    DEBUG = False
    _, _, rir, _ = forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(
                                        origin_sources, origin_mics,
                                        room2D, delays,
                                        mics_distances=None, phi_mics=0, 
                                        coords_mics=None, num_mics_circ_array=num_mics, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
                                        sources_distances=None, phi_sources_array=None, phi0_sources=0, mics_R=mics_R, 
                                        fs=fs, max_order=0,
                                        enable_prints=DEBUG, is_plot_room=False, room_plot_out='localization_plotted_room.pdf')
    print(f'Time: {time.time()-start_time}')
    max_len_rir = max([len(rir[i][j])
                        for i, j in product(range(num_mics), range(len(rir[0])))])
    

    rir = onp.array([[onp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir[j]] for j in range(len(rir))])
    onp.save(os.path.join(rir_dirs[0], 'direct_component.npy'), rir)

    # UNCOMMENT TO SAVE RIR BY ORDER (watch also function compute_rir_foreach_location)
    # for k in range(args['max_order']+1):
    #     rir_by_order = onp.array([[onp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir_by_orders[k][j]] for j in range(len(rir_by_orders[k]))])
    #     onp.save(os.path.join(rir_dirs[k+1], origin_mics_str+'.npy'), rir_by_order)

def process_first_order_image(data_in):
    # Compute the first order image with the given data
    xy, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, rir_dirs = data_in
    x = xy[0]
    y = xy[1]
    origin_sources = origin2D + onp.array([[x, y]])
    origin_mics = [onp.array([x,y])]
    # don't use round here for x,y! this will raise an error in the positions collection.
    print(f'Processing position: {origin_mics[0]}')
    origin_mics_str = f'{int(x*100)}_{int(y*100)}'
    start_time = time.time()
    # UNCOMMENT TO SAVE RIR BY ORDERS
    # _, _, rir, rir_by_orders = forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(
    #                                     origin_sources, origin_mics,
    #                                     room2D, delays,
    #                                     mics_distances=None, phi_mics=0, 
    #                                     coords_mics=None, num_mics_circ_array=num_mics, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
    #                                     sources_distances=None, phi_sources_array=None, phi0_sources=0, mics_R=mics_R, 
    #                                     fs=fs, max_order=args['max_order'],
    #                                     enable_prints=False, is_plot_room=True, room_plot_out='localization_plotted_room.pdf')
    DEBUG = False
    _, _, _, rir_by_orders = forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(
                                        origin_sources, origin_mics,
                                        room2D, delays,
                                        mics_distances=None, phi_mics=0, 
                                        coords_mics=None, num_mics_circ_array=num_mics, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
                                        sources_distances=None, phi_sources_array=None, phi0_sources=0, mics_R=mics_R, 
                                        fs=fs, max_order=1,
                                        enable_prints=DEBUG, is_plot_room=False, room_plot_out='localization_plotted_room.pdf')
    rir = rir_by_orders[1]
    print(f'Time: {time.time()-start_time}')
    max_len_rir = max([len(rir[i][j])
                        for i, j in product(range(num_mics), range(len(rir[0])))])
    

    rir = onp.array([[onp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir[j]] for j in range(len(rir))])
    onp.save(os.path.join(rir_dirs[0], origin_mics_str+'.npy'), rir)

    # UNCOMMENT TO SAVE RIR BY ORDER (watch also function compute_rir_foreach_location)
    # for k in range(args['max_order']+1):
    #     rir_by_order = onp.array([[onp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir_by_orders[k][j]] for j in range(len(rir_by_orders[k]))])
    #     onp.save(os.path.join(rir_dirs[k+1], origin_mics_str+'.npy'), rir_by_order)

def compute_first_order_image(args, delta=0.5, margins=0.02, cpu_pool=40):
    # Compute a few first order images
    print('Building the room')
    num_mics = args['num_phases_mics']
    num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)
    mics_R = 2*rotor_length_meter
    origin2D -= onp.array([[args['org_x'], args['org_y']]])
    
    # xx = onp.arange(mics_R+delta, args['room_x']-mics_R-delta, delta)
    # yy = onp.arange(mics_R+delta, args['room_y']-mics_R-delta, delta)
    xx = onp.arange( round((mics_R+margins), 2), round((args['room_x']-mics_R-margins),2), delta)
    yy = onp.arange( round((mics_R+margins), 2), round((args['room_y']-mics_R-margins),2), delta)
    

    # create an empty df to save the processed data
    #save_dir = os.path.join('/mnt', 'ssd','tomh', 'rir_data', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #save_dir = os.path.join('/mnt', 'walkure_public','tomh', 'rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    save_dir = os.path.join(cur_dir,'..','..','data','rir', f'first_order_{round(args["e_absorption"],2)}')
    os.makedirs(save_dir, exist_ok=True)
    #save_dir = os.path.join('/mnt', 'walkure_public','tomh', 'rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #rir_indoor_dir = os.path.join(save_dir, 'rir_indoor')
    
    # UNCOMMENT TO SAVE RIR BY ORDER (watch also function process_location) 
    # rir_dirs = [rir_indoor_dir] + [os.path.join(save_dir, 'rir_images'+str(k)) for k in range(args['max_order']+1)]

    # rir_dirs = [rir_indoor_dir]
    rir_dirs = [save_dir]
    for rir_dir_path in rir_dirs:
        if not os.path.exists(rir_dir_path):
            os.makedirs(rir_dir_path)
    

    xxyy = onp.array(onp.meshgrid(xx,yy)).T.reshape(-1,2)
    # skip previous-calculaed RIRs
    process_x = lambda x: (int(x.strip().split('.')[0].split('_')[0]), int(x.strip().split('.')[0].split('_')[1]))
    already_processed = [process_x(x) for x in sorted(os.listdir(rir_dirs[-1]))]
    xxyy = [x for x in xxyy if (int(x[0]*100), int(x[1]*100)) not in already_processed]
    
    print(f'Number of locations to process: {len(xxyy)}')

    data_inputs = [[xy, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, rir_dirs] for xy in xxyy]
    print('Computing RIR foreach location in the room')
    with Pool(cpu_pool) as pool:
        pool.map(process_first_order_image, data_inputs)

    # for data_in in data_inputs:
    #     process_location(data_in)

    return num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name

def process_location_and_angle(data_in):
    xyt, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, rir_dirs, compress_data_flag = data_in
    x = xyt[0]
    y = xyt[1]
    t = xyt[2]
    # breakpoint()
    origin_sources = origin2D + onp.array([[x, y]])
    origin_mics = [onp.array([x,y])]
    # don't use round here for x,y! this will raise an error in the positions collection.
    print(f'Processing position: {origin_mics[0]} with angle {t}')
    origin_mics_str = f'{int(x*10**8)}_{int(y*10**8)}_{int(t*10**8)}'
    start_time = time.time()
    # UNCOMMENT TO SAVE RIR BY ORDERS
    # _, _, rir, rir_by_orders = forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(
    #                                     origin_sources, origin_mics,
    #                                     room2D, delays,
    #                                     mics_distances=None, phi_mics=0, 
    #                                     coords_mics=None, num_mics_circ_array=num_mics, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
    #                                     sources_distances=None, phi_sources_array=None, phi0_sources=0, mics_R=mics_R, 
    #                                     fs=fs, max_order=args['max_order'],
    #                                     enable_prints=False, is_plot_room=True, room_plot_out='localization_plotted_room.pdf')
    DEBUG = False
    try:
        _, _, rir, _ = forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(
                                            origin_sources, origin_mics,
                                            room2D, delays,
                                            mics_distances=None, phi_mics=t, 
                                            coords_mics=None, num_mics_circ_array=num_mics, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
                                            sources_distances=None, phi_sources_array=None, phi0_sources=t, mics_R=mics_R, 
                                            fs=fs, max_order=args['max_order'],
                                            enable_prints=DEBUG, is_plot_room=False, room_plot_out='localization_plotted_room.pdf')
        print(f'Time: {time.time()-start_time}')
        max_len_rir = max([len(rir[i][j])
                            for i, j in product(range(num_mics), range(len(rir[0])))])
        

        rir = onp.array([[onp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir[j]] for j in range(len(rir))])
        if not compress_data_flag:
            onp.save(os.path.join(rir_dirs[0], origin_mics_str+'.npy'), rir)
        else:
            hdf5.save(os.path.join(rir_dirs[0], origin_mics_str+'.hdf5'), rir)
    except Exception as e:
        print(f'Time: {time.time()-start_time}')
        print(f"Could not compute the rir {origin_mics_str}. Skipping this data point (Got {e})")
        print(traceback.format_exc())
        # or
        print(sys.exc_info()[2])

    # UNCOMMENT TO SAVE RIR BY ORDER (watch also function compute_rir_foreach_location)
    # for k in range(args['max_order']+1):
    #     rir_by_order = onp.array([[onp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir_by_orders[k][j]] for j in range(len(rir_by_orders[k]))])
    #     onp.save(os.path.join(rir_dirs[k+1], origin_mics_str+'.npy'), rir_by_order)

def convert_coordinates(coordinates,n_samples, shear_angle=None, mics_R=None, margins=None, room_x=None,
                        min_coordinate_x=0.93, min_coordinate_y=0.93, max_coordinate_x=4.07, max_coordinate_y=4.07):
    assert len(coordinates.shape)==2 and coordinates.shape[1]==3, f"Coordinates must have shape (n,3). Got {coordinates}"
    assert n_samples<coordinates.shape[0], f"Too many samples asked. Asked {n_samples} from a pool of {coordinates.shape} samples"
    
    if shear_angle is not None:
        # get shearable coordinates
        assert mics_R is not None and margins is not None and room_x is not None, f"mics_R, margins and room_x can't be None but got {mics_R, margins,room_x}"
        shear_angle = onp.deg2rad(shear_angle)
        cotan = 1/onp.tan(shear_angle)
        # boundary = mics_R+margins
        # #coordinates = coordinates[boundary <= coordinates[:,0] -cotan * coordinates[:,1] <= room_x-boundary]
        # coordinates = coordinates[(coordinates[:,0]-cotan*coordinates[:,1] >= boundary) &
        #                           (coordinates[:,0]-cotan*coordinates[:,1] <= room_x-boundary)]
        coordinates = coordinates[(coordinates[:,0]-cotan*coordinates[:,1] >=0.0) &
                                  (coordinates[:,0]-cotan*coordinates[:,1] <=1.0)]
        feasible_coordinates = coordinates.shape[0]
        print(f"Number of feasible coordinates: {feasible_coordinates}")
        n_samples = min(n_samples, feasible_coordinates)
    
    sampled_coordinates = coordinates[onp.random.choice(coordinates.shape[0], size=(n_samples), replace=False)]
    sampled_coordinates[:, 0] = sampled_coordinates[:,0]*(max_coordinate_x-min_coordinate_x)+min_coordinate_x
    sampled_coordinates[:, 1] = sampled_coordinates[:,1]*(max_coordinate_y-min_coordinate_y)+min_coordinate_y

    if shear_angle is not None:
        # apply shear
        sampled_coordinates[:,0] = sampled_coordinates[:,0]+cotan*sampled_coordinates[:,1]
    return sampled_coordinates

def get_cluster_positions(origin: onp.ndarray, grid_side: int = 3, grid_step: float=0.05, number_of_angles: int = 90):
    radius: int = grid_side//2
    starting_coordinates: onp.ndarray = origin-(grid_step*radius)
    epsilon = 1e-4
    ending_coordinates: onp.ndarray = origin+(grid_step*radius)+epsilon
    xx: onp.ndarray = onp.arange(starting_coordinates[0], ending_coordinates[0], grid_step)
    yy: onp.ndarray = onp.arange(starting_coordinates[1], ending_coordinates[1], grid_step)
    xxyy: onp.ndarray = onp.array(onp.meshgrid(xx,yy)).T.reshape(-1,2)
    number_of_points: int = xxyy.shape[0]
    xxyy = onp.repeat(xxyy, number_of_angles, 0)
    xxyy = onp.round(xxyy,2)

    tt: onp.ndarray = onp.linspace(0, 2*onp.pi, number_of_angles)
    tt = onp.expand_dims(tt, 0)
    tt = onp.tile(tt, (number_of_points, 1))
    tt = tt.reshape(-1, 1)

    xxyytt: onp.ndarray = onp.concatenate((xxyy,tt), axis=-1)
    return xxyytt

def get_small_dense_room_samples(args, mics_R):
    step = 0.1
    offset = 0.02
    number_of_angles = 64
    margins = 0.02
    

    x0,x1 = round((mics_R+margins), 2)+offset, round((args['room_x']-mics_R-margins),2)-offset
    y0,y1 = round((mics_R+margins), 2)+offset, round((args['room_y']-mics_R-margins),2)-offset

    xx = onp.arange(x0 , x1, step)
    yy = onp.arange(y0, y1, step)
    xxyy = onp.array(onp.meshgrid(xx,yy)).T.reshape(-1,2)
    number_of_points = xxyy.shape[0]
    xxyy = onp.repeat(xxyy, number_of_angles, 0)

    tt = onp.linspace(0, 2*onp.pi, number_of_angles, endpoint=False)
    tt = onp.tile(tt, number_of_points)
    tt = onp.expand_dims(tt,1)

    xxyytt = onp.concatenate((xxyy, tt), axis=-1)
    return xxyytt

def compute_rir_foreach_location_and_angle(args, delta=0.02, delta_angle=onp.pi/8, margins=0.02, cpu_pool=40, by_demand=False,
                                           number_of_angles=16, num_points_per_side=None, rotating_angle=None):
    print('Building the room')
    num_mics = args['num_phases_mics']
    num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)
    mics_R = 2*rotor_length_meter
    origin2D -= onp.array([[args['org_x'], args['org_y']]])
    #breakpoint()
    # xx = onp.arange(mics_R+delta, args['room_x']-mics_R-delta, delta)
    # yy = onp.arange(mics_R+delta, args['room_y']-mics_R-delta, delta)
    # xx = onp.arange( round((mics_R+margins), 2), round((args['room_x']-mics_R-margins),2), delta)
    # yy = onp.arange( round((mics_R+margins), 2), round((args['room_y']-mics_R-margins),2), delta)
    # tt = onp.arange(0,2*onp.pi, step=delta_angle)
    if args['use_small_dense_room']:
        xxyytt = get_small_dense_room_samples(args, mics_R)
    elif args['cluster_around_point']:
        assert len(args['cluster_center']) == 2, f"Expected the 2d coordinates of the center of the cluster, but got {args['cluster_center']}"
        cluster_center: onp.ndarray = onp.array(args['cluster_center'])
        xxyytt = get_cluster_positions(cluster_center)
    elif len(args['corners']) > 0:
        distance_from_wall: float = round(mics_R+margins,2)+args['extra_distance_from_wall']
        corners = onp.array(args['corners']).reshape(-1,2)
        non_convex_offset = args['non_convex_offset']
        xxyy = get_points_in_polygon_from_corners(corners, num_points_per_side, distance_from_wall, non_convex_offset)
        num_points = xxyy.shape[0]
        xxyy = onp.repeat(xxyy, number_of_angles, 0)
        if number_of_angles < 32:
            seed: int = 0
            tt = onp.random.default_rng(seed).uniform(0, 2*onp.pi, size=((num_points)*number_of_angles))
            #tt = PoissonDisk(d=1, radius=0.03, ncandidates=360, seed=seed).random(n=)
        else:
            tt = onp.linspace(0, 2*onp.pi, number_of_angles)
            tt = onp.tile(tt, (num_points))
        xxyytt = onp.concatenate((xxyy, onp.expand_dims(tt, axis=-1)), axis=-1).reshape(-1,3)
    elif args['use_relative_units']:
        reference_coordinates = onp.load(args['reference_coordinates_path'])
        if rotating_angle is None:
            min_coordinate_x, min_coordinate_y = round((mics_R+margins), 2), round((mics_R+margins), 2)
            max_coordinate_x, max_coordinate_y = round((args['room_x']-mics_R-margins),2), round((args['room_y']-mics_R-margins),2)
            xxyytt = convert_coordinates(reference_coordinates,args['n_samples'],shear_angle=None,
                                         min_coordinate_x=min_coordinate_x,min_coordinate_y=min_coordinate_y,
                                         max_coordinate_x=max_coordinate_x,max_coordinate_y=max_coordinate_y)
        else:
            cotan_theta = 1/onp.tan(onp.deg2rad(rotating_angle))
            min_coordinate_x = round(cotan_theta*onp.sqrt(1+1/cotan_theta**2)*(mics_R+margins), 2)
            min_coordinate_y = round((mics_R+margins), 2)

            max_coordinate_x = round(args['room_x'] - cotan_theta*onp.sqrt(1+1/cotan_theta**2)*(mics_R+margins), 2)
            max_coordinate_y = round((args['room_y']-mics_R-margins),2)

            xxyytt = convert_coordinates(reference_coordinates,args['n_samples'],rotating_angle,
                                         mics_R=mics_R,margins=margins,room_x=args['room_x'],
                                         min_coordinate_x=min_coordinate_x, min_coordinate_y=min_coordinate_y,
                                         max_coordinate_x=max_coordinate_x,max_coordinate_y=max_coordinate_y)
        
    elif num_points_per_side is None:
        xx = onp.arange( round((mics_R+margins), 2), round((args['room_x']-mics_R-margins),2), delta)
        yy = onp.arange( round((mics_R+margins), 2), round((args['room_y']-mics_R-margins),2), delta)
    else:
        if rotating_angle is not None:
            cotan_theta = 1/onp.tan(onp.deg2rad(rotating_angle))

            # y_min = round((mics_R+margins), 2)
            # x_min = round(cotan_theta*(y_min + onp.sqrt(1+1/cotan_theta**2)*(mics_R+margins)) - cotan_theta*y_min, 2)

            # y_max = round((args['room_y']-mics_R-margins),2)
            # x_max = round(cotan_theta*(y_min + 1/cotan_theta*args['room_x']-onp.sqrt(1+1/cotan_theta**2)*(mics_R+margins)) - cotan_theta*y_min, 2)
            min_coordinate_x = round(cotan_theta*onp.sqrt(1+1/cotan_theta**2)*(mics_R+margins), 2)
            min_coordinate_y = round((mics_R+margins), 2)

            max_coordinate_x = round(args['room_x'] - cotan_theta*onp.sqrt(1+1/cotan_theta**2)*(mics_R+margins), 2)
            max_coordinate_y = round((args['room_y']-mics_R-margins),2)
            
            xx = onp.linspace(min_coordinate_x, max_coordinate_x, num_points_per_side)
            yy = onp.linspace(min_coordinate_y, max_coordinate_y, num_points_per_side)

            xxyy = onp.array(onp.meshgrid(xx,yy)).T.reshape(-1,2)
            xxyy = shear(xxyy, rotating_angle)
            xxyy = onp.round(xxyy,2)

            # xxyy = onp.tile(xxyy, (number_of_angles,1))
            xxyy = onp.repeat(xxyy, number_of_angles, 0)

            if number_of_angles < 32:
                seed: int = 0
                tt = onp.random.default_rng(seed).uniform(0, 2*onp.pi, size=((num_points_per_side**2)*number_of_angles))
            else:
                tt = onp.linspace(0, 2*onp.pi, number_of_angles, endpoint=False)
                tt = onp.tile(tt, (num_points_per_side**2))

            xxyytt = onp.concatenate((xxyy, onp.expand_dims(tt, axis=-1)), axis=-1).reshape(-1,3)
            #breakpoint()

        else:
            xx = onp.linspace( round((mics_R+margins), 2), round((args['room_x']-mics_R-margins),2), num_points_per_side)
            yy = onp.linspace( round((mics_R+margins), 2), round((args['room_y']-mics_R-margins),2), num_points_per_side)

            # seed: int = 0
            # tt = onp.random.default_rng(seed).uniform(0, 2*onp.pi, size=(len(xx), len(yy), number_of_angles))
            # # UNCOMMENT to force to include the angles 0, pi/2, pi, 3pi/2
            # if number_of_angles >= 4:
            #     tt[..., -4:] = onp.array([[0, onp.pi/2, onp.pi, 3*onp.pi/2]] * (len(xx) * len(yy))).reshape(len(xx), len(yy), 4)

            if args['use_newton_cluster']:
                # user = args['user'] # getuser()
                # angle_offset = 0 if user == 'tamir.shor' else 1
                # print(f"Recognized the user {user}: using offset {angle_offset}")
                # if number_of_angles == 64:
                #     # exploit the previously computed angles
                #     tt = onp.linspace(0, 2*onp.pi, number_of_angles)[1::2][angle_offset::2]
                # else:
                #     tt = onp.linspace(0, 2*onp.pi, number_of_angles)[angle_offset::2]
                tt = onp.linspace(0, 2*onp.pi, number_of_angles, endpoint=False)
            else:
                if number_of_angles >= 32:
                    tt = onp.linspace(0, 2*onp.pi, number_of_angles)
                else:
                    seed: int = 0
                    tt = onp.random.default_rng(seed).uniform(0, 2*onp.pi, size=((num_points_per_side**2)*number_of_angles))
            
            if tt.shape[0] != (num_points_per_side**2)*number_of_angles:
                # this branch is taken in all of the cases besides when tt is sampled randomly 
                
                if args['use_newton_cluster']:
                    # if number_of_angles == 64:
                    #     xx = onp.tile(xx[..., None], (1, 1, (number_of_angles//4)))
                    #     yy = onp.tile(yy[..., None], (1, 1, (number_of_angles//4)))
                    # else:
                    #     xx = onp.tile(xx[..., None], (1, 1, (number_of_angles//2)))
                    #     yy = onp.tile(yy[..., None], (1, 1, (number_of_angles//2)))
                    xx, yy = onp.meshgrid(xx, yy)
                    xxyy = onp.array([xx,yy]).T.reshape(-1,2)
                    number_of_points = xxyy.shape[0]
                    xxyy = onp.repeat(xxyy, number_of_angles, axis=0)
                    tt = onp.tile(tt, number_of_points)
                    tt = onp.expand_dims(tt, 1)
                    xxyytt = onp.concatenate((xxyy,tt), axis=-1)
                else:
                    tt = tt.reshape(1,1,-1)
                    tt = onp.tile(tt, (num_points_per_side, num_points_per_side, 1))
                    xx, yy = onp.meshgrid(xx, yy)
                    xx = onp.tile(xx[..., None], (1, 1, (number_of_angles)))
                    yy = onp.tile(yy[..., None], (1, 1, (number_of_angles)))
                    xxyytt = onp.stack((xx, yy, tt), axis=-1).reshape(-1, 3)
            else:
                xxyy = onp.array(onp.meshgrid(xx,yy)).T.reshape(-1,2)
                xxyy = onp.tile(xxyy, (number_of_angles,1))
                xxyytt = onp.concatenate((xxyy, onp.expand_dims(tt, axis=-1)), axis=-1).reshape(-1,3)
    


    # create an empty df to save the processed data
    #save_dir = os.path.join('/mnt', 'ssd','tomh', 'rir_data', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #save_dir = os.path.join('/mnt', 'walkure_public','tomh', 'rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #save_dir = os.path.join(cur_dir,'..','..','data','rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}_d_angle_{delta_angle}')
    
    #/mnt/walkure_public/tamirs/
    #save_dir = os.path.join('/mnt','walkure_public', 'tamirs', 'rir_orientation_angle', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}_d_angle_{delta_angle}')
    if args['use_small_dense_room']:
        save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'small_dense_room', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
    elif args['cluster_around_point']:
        if args['use_newton_cluster']:
            save_dir = os.path.join('/', 'home', 'gabrieles', 'EARS', 'data', 'pressure_field_orientation_dataset', 'cluster_point', f"{args['cluster_center'][0]}_{args['cluster_center'][1]}", f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
        else:
            save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'cluster_point', f"{args['cluster_center'][0]}_{args['cluster_center'][1]}", f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
    elif len(args['corners']) > 0:
        folder_name = 'non_convex_room' if args['folder_name'] is None else args['folder_name']
        if args['use_newton_cluster']:
            save_dir = os.path.join('/', 'home', 'gabrieles', 'EARS', 'data', 'pressure_field_orientation_dataset', folder_name, f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
        else:
            save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', folder_name, f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
        print(f"Data will be saved in {save_dir}")
    elif args['use_newton_cluster']:
        # save_dir = os.path.join('/', 'home', 'gabrieles', 'ears', 'code', 'data', 'pressure_field_orientation_dataset', f'{number_of_angles}_angles', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
        user = args['user'] # getuser()
        if args['folder_name'] is not None:
            folder_name = args['folder_name']
        elif number_of_angles != 64:
            folder_name = f'{number_of_angles}_angles'
        else:
            folder_name = '32_angles'
        #folder_name = f'{number_of_angles}_angles' if number_of_angles != 64 else '32_angles'
        save_dir = os.path.join('/', 'home', user, 'EARS', 'data', 'pressure_field_orientation_dataset', folder_name, f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
    elif args['folder_name'] is not None:
        save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', args['folder_name'], f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')         
    elif args['saved_data_dir_path'].startswith('rel_units') or args['saved_data_dir_path'].startswith('consistent') or args['saved_data_dir_path'].startswith('rir_shear') or args['saved_data_dir_path'].startswith('rir_uniform'):
        # TO CHANGE: Uncomment this
        save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'robustness_test', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')     
    else:
        # TO CHANGE: Uncomment this
        #save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'big_dataset', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
        if args['save_in_mega']:
            save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'mega_dataset', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
        else:
            # save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', f'{number_of_angles}_dataset', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
            save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')


    os.makedirs(save_dir, exist_ok=True)
    #save_dir = os.path.join('/mnt', 'walkure_public','tomh', 'rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    rir_indoor_dir = os.path.join(save_dir, 'rir_indoor')
    
    # UNCOMMENT TO SAVE RIR BY ORDER (watch also function process_location) 
    # rir_dirs = [rir_indoor_dir] + [os.path.join(save_dir, 'rir_images'+str(k)) for k in range(args['max_order']+1)]

    rir_dirs = [rir_indoor_dir]
    for rir_dir_path in rir_dirs:
        if not os.path.exists(rir_dir_path):
            os.makedirs(rir_dir_path)
    
    # check if there is the flag corrupted or not: 
            # if so just compute the corrupted ones and skip process_x and already_processed, but rather set xxyytt to these coordinates
            # otherwise just do process_x, already_processed and xxyytt as always
    #if args['fix_corrupted']:
    #to_fix = [i.removesuffix('.corrupted') for i in os.listdir(save_dir) if i.endswith('.corrupted')]
    #coordinates_to_fix = [tuple(map(int, i.split('_'))) for i in to_fix]
        #xxyytt = [datapoint for datapoint in xxyytt if (int(datapoint[0]*10**8), int(datapoint[1]*10**8), int(datapoint[2]*10**8)) in coordinates]

    #else:
        #xxyytt = onp.array(onp.meshgrid(xx,yy,tt)).T.reshape(-1,3)
        # skip previous-calculaed RIRs
    process_x = lambda x: (
        int(x.strip().split('.')[0].split('_')[0]), 
        int(x.strip().split('.')[0].split('_')[1]), 
        int(x.strip().split('.')[0].split('_')[2])
    )
    #precomputed_rir_dir = "/datasets/rir_indoor"
    #is_valid_file = lambda root : lambda x: os.path.isfile(os.path.join(root, x)) and not x.endswith('.txt')
    # computed_files = set(filter(lambda x: '.' in x and not x.endswith('.txt'), os.listdir(rir_dirs[-1])))
    computed_files = set(filter(lambda x: '.' in x and not x.endswith('.txt'), get_listed_files(rir_dirs[-1])))
    #computed_files = set(filter(is_valid_file(rir_dirs[-1]), get_listed_files(rir_dirs[-1])))


    #computed_files = set(filter(is_valid_file(rir_dirs[-1]), os.listdir(rir_dirs[-1])))
    print(f"We have already computed {len(computed_files)} RIRs that are in this directory ({rir_dirs[-1]})")

    corrupted_files = set(filter(lambda x: x.endswith('.corrupted'), computed_files))
    print(f"There are {len(corrupted_files)} corrupted files. Removing them...")

    # clean corrupted files
    for corrupted_file in corrupted_files:
        try:
            os.remove(os.path.join(rir_dirs[-1], corrupted_file))
        except Exception as e:
            print(f"Could not remove {os.path.join(rir_dirs[-1], corrupted_file)} . Got {e} . Skipping its removal")
    print(f"{len(corrupted_files)} corrupted files removed")

    corrupted_files = computed_files.difference(corrupted_files)

    # sound_folder: str = os.listdir(os.path.join(rir_dirs[-1], '..', '..'))
    sound_folder: str = get_listed_files(os.path.join(rir_dirs[-1], '..', '..'))


    computed_sound = set(filter(lambda x:  '.' in x and not x.endswith('.txt'), sound_folder))
    #computed_sound = set(filter(is_valid_file(os.path.join(rir_dirs[-1], '..', '..')), sound_folder))
    # erase any RIR whose corresponding sound has already been computed:
    for redundant_file in computed_sound.intersection(computed_files):
        try:
            os.remove(os.path.join(rir_dirs[-1], redundant_file))
        except:
            print(f"Couldn't remove {os.path.join(rir_dirs[-1], redundant_file)}. Probably another process removed it in the meantime")

    
    file_list_path: str = os.path.join(rir_dirs[-1], '..', '..', 'file_list.txt')
    if os.path.exists(file_list_path):
        with open(file_list_path, "r") as f:
            computed_sound_in_other_server = set(filter(lambda x:x.endswith('.hdf5\n') or x.endswith('.hdf5'), f.readlines()))
        computed_sound = computed_sound.union(computed_sound_in_other_server)

    computed_files = computed_files.union(computed_sound)

    # extension = ".npy" if not args['compress_data'] else ".hdf5"
    # computed_files = set(filter(lambda x:x.endswith(extension), os.listdir(rir_dirs[-1])))
    number_filter = lambda x: x.strip().split('.')[0].split('_')[0].isdigit() and x.strip().split('.')[0].split('_')[1].isdigit() \
                              and x.strip().split('.')[0].split('_')[2].isdigit()
    computed_files = set(filter(number_filter, computed_files))
    already_processed = set([process_x(x) for x in computed_files])


    #already_processed = [process_x(x) for x in sorted(os.listdir(precomputed_rir_dir))]
    # xxyytt = sorted([x for x in xxyytt if (int(x[0]*10**8), int(x[1]*10**8), int(x[2]*10**8)) not in already_processed])
    xxyytt = sorted(list(
        filter(lambda x: (int(x[0]*10**8), int(x[1]*10**8), int(x[2]*10**8)) not in already_processed, xxyytt.tolist()
               )))
    
    print(f'Number of locations to process: {len(xxyytt)}')
    

    data_inputs = [[xyt, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, rir_dirs, args['compress_data']] for xyt in xxyytt]
    print('Computing RIR foreach location in the room')
    
    if args['generation_computation'] == 'concurrent':
        print(f"Generating data concurrently using {cpu_pool} cores")
        with Pool(cpu_pool) as pool:
            pool.map(process_location_and_angle, data_inputs)
    elif args['generation_computation'] == 'iterative':
        print("Generating data iteratively")
        for data_in in data_inputs:
            process_location_and_angle(data_in)
    else:
        raise ValueError("Please set -generation_computation either to concurrent or to iterative")

    return num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name

def process_location(data_in):
    xy, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, rir_dirs = data_in
    x = xy[0]
    y = xy[1]
    # origin2D stores the displacement of the rotors and we need to place the sources accordingly
    origin_sources = origin2D + onp.array([[x, y]])
    # while the microphones need to be kept at the center of mass of the drone
    origin_mics = [onp.array([x,y])]
    # don't use round here for x,y! this will raise an error in the positions collection.
    print(f'Processing position: {origin_mics[0]}')
    origin_mics_str = f'{int(x*100)}_{int(y*100)}'
    start_time = time.time()
    # UNCOMMENT TO SAVE RIR BY ORDERS
    # _, _, rir, rir_by_orders = forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(
    #                                     origin_sources, origin_mics,
    #                                     room2D, delays,
    #                                     mics_distances=None, phi_mics=0, 
    #                                     coords_mics=None, num_mics_circ_array=num_mics, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
    #                                     sources_distances=None, phi_sources_array=None, phi0_sources=0, mics_R=mics_R, 
    #                                     fs=fs, max_order=args['max_order'],
    #                                     enable_prints=False, is_plot_room=True, room_plot_out='localization_plotted_room.pdf')
    DEBUG = True

    '''
    ARGUMENTS FOR 0.93, 0.93:

    origin_sources: array([[0.5871, 1.2729],
       [0.5871, 1.2729],
       [0.5871, 0.5871],
       [0.5871, 0.5871],
       [1.2729, 1.2729],
       [1.2729, 1.2729],
       [1.2729, 0.5871],
       [1.2729, 0.5871]]) CHANGES ACROSS DATAPOINTS
    origin_mics: [array([0.93, 0.93])] CHANGES ACROSS DATAPOINTS
    room2D: {'obstructing_walls': array([], dtype=int32), 'dim': 2, 't0': 0.013316751298034278, 'fs': 3003.735603735763, 'absorption': array([0.29289322, 0.29289322, 0.29289322, 0.29289322]), 'normals': array([[ 1.,  0., -1.,  0.],
       [-0.,  1., -0., -1.]]), 'corners': array([[5., 5., 0., 0.],
       [0., 5., 5., 0.]], dtype=float32), 'walls': [{'corners': array([[5., 5.],
       [0., 5.]], dtype=float32), 'normal': array([ 1., -0.]), 'dim': 2, 'absorption': 0.2928932188134524, 'name': None}, {'corners': array([[5., 0.],
       [5., 5.]], dtype=float32), 'normal': array([0., 1.]), 'dim': 2, 'absorption': 0.2928932188134524, 'name': None}, {'corners': array([[0., 0.],
       [5., 0.]], dtype=float32), 'normal': array([-1., -0.]), 'dim': 2, 'absorption': 0.2928932188134524, 'name': None}, {'corners': array([[0., 5.],
       [0., 0.]], dtype=float32), 'normal': array([ 0., -1.]), 'dim': 2, 'absorption': 0.2928932188134524, 'name': None}]}
    delays: array([0., 0., 0., ..., 0., 0., 0.]) (shape is (1024,))
    num_mics: 8
    radiuses_circles: array([0.2286, 0.51  , 0.2286, 0.51  , 0.2286, 0.51  , 0.2286, 0.51  ])
    num_sources: 128
    mics_R: 0.9144
    fs: 3003.735603735763
    args['max_order']: 1

    ARGUMENTS FOR 4.07, 0.93:

    origin_sources: array([[3.7271, 1.2729],
       [3.7271, 1.2729],
       [3.7271, 0.5871],
       [3.7271, 0.5871],
       [4.4129, 1.2729],
       [4.4129, 1.2729],
       [4.4129, 0.5871],
       [4.4129, 0.5871]]) CHANGES ACROSS DATAPOINTS
    origin_mics: [array([4.07, 0.93])] CHANGES ACROSS DATAPOINTS
    {'obstructing_walls': array([], dtype=int32), 'dim': 2, 't0': 0.013316751298034278, 'fs': 3003.735603735763, 'absorption': array([0.29289322, 0.29289322, 0.29289322, 0.29289322]), 'normals': array([[ 1.,  0., -1.,  0.],
       [-0.,  1., -0., -1.]]), 'corners': array([[5., 5., 0., 0.],
       [0., 5., 5., 0.]], dtype=float32), 'walls': [{'corners': array([[5., 5.],
       [0., 5.]], dtype=float32), 'normal': array([ 1., -0.]), 'dim': 2, 'absorption': 0.2928932188134524, 'name': None}, {'corners': array([[5., 0.],
       [5., 5.]], dtype=float32), 'normal': array([0., 1.]), 'dim': 2, 'absorption': 0.2928932188134524, 'name': None}, {'corners': array([[0., 0.],
       [5., 0.]], dtype=float32), 'normal': array([-1., -0.]), 'dim': 2, 'absorption': 0.2928932188134524, 'name': None}, {'corners': array([[0., 5.],
       [0., 0.]], dtype=float32), 'normal': array([ 0., -1.]), 'dim': 2, 'absorption': 0.2928932188134524, 'name': None}]}
    delays: array([0., 0., 0., ..., 0., 0., 0.]) (shape is (1024,))
    num_mics: 8
    radiuses_circles: array([0.2286, 0.51  , 0.2286, 0.51  , 0.2286, 0.51  , 0.2286, 0.51  ])
    num_sources: 128
    mics_R: 0.9144
    fs: 3003.735603735763
    args['max_order']: 1
    '''

    _, _, rir, _ = forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(
                                        origin_sources, origin_mics,
                                        room2D, delays,
                                        mics_distances=None, phi_mics=0, 
                                        coords_mics=None, num_mics_circ_array=num_mics, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
                                        sources_distances=None, phi_sources_array=None, phi0_sources=0, mics_R=mics_R, 
                                        fs=fs, max_order=args['max_order'],
                                        enable_prints=DEBUG, is_plot_room=DEBUG, room_plot_out='localization_plotted_room.pdf')
    print(f'Time: {time.time()-start_time}')
    max_len_rir = max([len(rir[i][j])
                        for i, j in product(range(num_mics), range(len(rir[0])))])
    

    rir = onp.array([[onp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir[j]] for j in range(len(rir))])
    onp.save(os.path.join(rir_dirs[0], origin_mics_str+'.npy'), rir)

    # UNCOMMENT TO SAVE RIR BY ORDER (watch also function compute_rir_foreach_location)
    # for k in range(args['max_order']+1):
    #     rir_by_order = onp.array([[onp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir_by_orders[k][j]] for j in range(len(rir_by_orders[k]))])
    #     onp.save(os.path.join(rir_dirs[k+1], origin_mics_str+'.npy'), rir_by_order)

def shear(xxyy, angle):
    '''
    Shear a set of points by angle in degrees around the origin
    :param xxyy: set of points stored in shape (num_points, 2)
    :param angle: the rotating angle in degrees
    :return rotated points
    '''
    angle = onp.deg2rad(angle)
    c = 1/onp.tan(angle)
    R = onp.array([
        [1, c],
        [0, 1],
        ])  
    return (R@(xxyy.T)).T

def get_line_x(p1,p2,y):
        x1,y1 = p1
        x2,y2 = p2
        return (x2-x1)/(y2-y1)*(y-y1+(y2-y1)/(x2-x1)*x1)

def compute_rir_foreach_location(args, delta=0.02, margins=0.02, cpu_pool=40, by_demand=False,
                                 num_points_per_side=None, rotating_angle=None):
    print('Building the room')
    num_mics = args['num_phases_mics']
    num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)
    mics_R = 2*rotor_length_meter
    origin2D -= onp.array([[args['org_x'], args['org_y']]])
    
    # xx = onp.arange(mics_R+delta, args['room_x']-mics_R-delta, delta)
    # yy = onp.arange(mics_R+delta, args['room_y']-mics_R-delta, delta)
    # mics_r = 0.9144; margins = 0.02; delta = 0.05
    if num_points_per_side is None:
        xx = onp.arange( round((mics_R+margins), 2), round((args['room_x']-mics_R-margins),2), delta)
        yy = onp.arange( round((mics_R+margins), 2), round((args['room_y']-mics_R-margins),2), delta)
    else:
        if rotating_angle is not None:
            cotan_theta = 1/onp.tan(onp.deg2rad(rotating_angle))

            y_min = round((mics_R+margins), 2)
            x_min = round(cotan_theta*(y_min + 1/cotan_theta*args['room_x']-onp.sqrt(1+1/cotan_theta**2)*(mics_R+margins)) - cotan_theta*y_min, 2)

            y_max = round((args['room_y']-mics_R-margins),2)
            x_max = round(cotan_theta*(y_min + onp.sqrt(1+1/cotan_theta**2)*(mics_R+margins)) - cotan_theta*y_min, 2)
            
            xx = onp.linspace(x_min, x_max, num_points_per_side)
            yy = onp.linspace(y_min, y_max, num_points_per_side)
        else:
            xx = onp.linspace( round((mics_R+margins), 2), round((args['room_x']-mics_R-margins),2), num_points_per_side)
            yy = onp.linspace( round((mics_R+margins), 2), round((args['room_y']-mics_R-margins),2), num_points_per_side)
    

    # create an empty df to save the processed data
    #save_dir = os.path.join('/mnt', 'ssd','tomh', 'rir_data', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #save_dir = os.path.join('/mnt', 'walkure_public','tomh', 'rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #save_dir = os.path.join(cur_dir,'..','..','data','rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    save_dir = os.path.join('/mnt', 'walkure_public','tamirs','rir2d_circular_boundary', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #save_dir = os.path.join('/mnt', 'walkure_public','tamirs','rir_robustness_test', 'uniform_deformation', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #save_dir = os.path.join('/mnt', 'walkure_public','tamirs','rir_robustness_test', 'absorption_coefficient', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    # save_dir = os.path.join('/mnt', 'walkure_public','tamirs','rir_robustness_test', 'shear', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #save_dir = os.path.join('/mnt', 'walkure_public','tamirs','rir_robustness_test', 'aspect_ratio', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')


    os.makedirs(save_dir, exist_ok=True)
    #save_dir = os.path.join('/mnt', 'walkure_public','tomh', 'rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    rir_indoor_dir = os.path.join(save_dir, 'rir_indoor')
    
    # UNCOMMENT TO SAVE RIR BY ORDER (watch also function process_location) 
    # rir_dirs = [rir_indoor_dir] + [os.path.join(save_dir, 'rir_images'+str(k)) for k in range(args['max_order']+1)]

    rir_dirs = [rir_indoor_dir]
    for rir_dir_path in rir_dirs:
        if not os.path.exists(rir_dir_path):
            os.makedirs(rir_dir_path)
    

    xxyy = onp.array(onp.meshgrid(xx,yy)).T.reshape(-1,2)

    if rotating_angle is not None:
        xxyy = shear(xxyy, rotating_angle)
        xxyy = onp.round(xxyy,2)

    # skip previous-calculaed RIRs
    process_x = lambda x: (int(x.strip().split('.')[0].split('_')[0]), int(x.strip().split('.')[0].split('_')[1]))
    already_processed = [process_x(x) for x in sorted(os.listdir(rir_dirs[-1]))]
    # TO UNCOMMENT!
    xxyy = [x for x in xxyy if (int(x[0]*100), int(x[1]*100)) not in already_processed]
    
    print(f'Number of locations to process: {len(xxyy)}')
    if by_demand:
        xxyy = [(2.5,0.93), (0.93,0.93), (2.5,2.5)]

    data_inputs = [[xy, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, rir_dirs] for xy in xxyy]
    print('Computing RIR foreach location in the room')
    # with Pool(cpu_pool) as pool:
    #     pool.map(process_location, data_inputs)

    for data_in in data_inputs:
        process_location(data_in)

    return num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name

def compute_rir_foreach_location_shear_deformation(args, delta=0.02, margins=0.02, cpu_pool=40, by_demand=False):
    raise NotImplementedError()
    print('Building the room')
    num_mics = args['num_phases_mics']
    num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)
    mics_R = 2*rotor_length_meter
    origin2D -= onp.array([[args['org_x'], args['org_y']]])
    
    # xx = onp.arange(mics_R+delta, args['room_x']-mics_R-delta, delta)
    # yy = onp.arange(mics_R+delta, args['room_y']-mics_R-delta, delta)
    xx = onp.arange( round((mics_R+margins), 2), round((args['room_x']-mics_R-margins),2), delta)
    yy = onp.arange( round((mics_R+margins), 2), round((args['room_y']-mics_R-margins),2), delta)
    

    # create an empty df to save the processed data
    #save_dir = os.path.join('/mnt', 'ssd','tomh', 'rir_data', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #save_dir = os.path.join('/mnt', 'walkure_public','tomh', 'rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    save_dir = os.path.join(cur_dir,'..','..','data','rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    os.makedirs(save_dir, exist_ok=True)
    #save_dir = os.path.join('/mnt', 'walkure_public','tomh', 'rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    rir_indoor_dir = os.path.join(save_dir, 'rir_indoor')
    
    # UNCOMMENT TO SAVE RIR BY ORDER (watch also function process_location) 
    # rir_dirs = [rir_indoor_dir] + [os.path.join(save_dir, 'rir_images'+str(k)) for k in range(args['max_order']+1)]

    rir_dirs = [rir_indoor_dir]
    for rir_dir_path in rir_dirs:
        if not os.path.exists(rir_dir_path):
            os.makedirs(rir_dir_path)
    

    xxyy = onp.array(onp.meshgrid(xx,yy)).T.reshape(-1,2)
    # skip previous-calculaed RIRs
    process_x = lambda x: (int(x.strip().split('.')[0].split('_')[0]), int(x.strip().split('.')[0].split('_')[1]))
    already_processed = [process_x(x) for x in sorted(os.listdir(rir_dirs[-1]))]
    xxyy = [x for x in xxyy if (int(x[0]*100), int(x[1]*100)) not in already_processed]
    
    print(f'Number of locations to process: {len(xxyy)}')
    if by_demand:
        xxyy = [(2.5,0.93), (0.93,0.93), (2.5,2.5)]

    data_inputs = [[xy, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, rir_dirs] for xy in xxyy]
    print('Computing RIR foreach location in the room')
    with Pool(cpu_pool) as pool:
        pool.map(process_location, data_inputs)

    # for data_in in data_inputs:
    #     process_location(data_in)

    return num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name

def simulate_signals_with_rir(args, delta=0.02, num_revolutions=4):

    #assert args['phase_modulation_path'] is not None, "Please specify which phase modulation to use by setting the flag -phase_modulation_path"

    num_mics = args['num_phases_mics']
    num_rotors = args['num_rotors']

    #dir_save_path = os.path.join('/mnt', 'walkure_public', 'tamirs', 'pressure_field_2d_no_padding',f'indoor_recordings_{num_rotors}_rotors_{num_mics}_mics_d_{delta}_mode_{args["indoor_mode"]}_{args["phase_modulation_path"]}')
    # dir_save_path = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'robustness_test', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    
    storage_location = 'walkure_public' # walkure_public
    if args['use_small_dense_room']:
        dir_save_path = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'small_dense_room', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    elif args['cluster_around_point']:
        if args['use_newton_cluster']:
            dir_save_path = os.path.join('/', 'home', 'gabrieles', 'EARS', 'data', 'pressure_field_orientation_dataset', 'cluster_point', f"{args['cluster_center'][0]}_{args['cluster_center'][1]}", f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
        else:
            dir_save_path = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'cluster_point', f"{args['cluster_center'][0]}_{args['cluster_center'][1]}", f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    elif len(args['corners']) > 0:
        folder_name = 'non_convex_room' if args['folder_name'] is None else args['folder_name']
        if args['use_newton_cluster']:
            dir_save_path = os.path.join('/', 'home', 'gabrieles', 'EARS', 'data', 'pressure_field_orientation_dataset', folder_name, f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
        else:
            dir_save_path = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', folder_name, f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
        print(f"Data will be saved in {dir_save_path}")
    elif args['use_newton_cluster']:
        user = args['user'] # getuser()
        # folder_name = f"{args['number_of_angles']}_angles" if args['number_of_angles'] != 64 else '32_angles'
        if args['folder_name'] is not None:
            folder_name = args['folder_name']
        elif args['number_of_angles'] != 64:
            folder_name = f"{args['number_of_angles']}_angles"
        else:
            folder_name = '32_angles'
        dir_save_path = os.path.join('/', 'home', user, 'EARS', 'data', 'pressure_field_orientation_dataset', folder_name, f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    elif args['folder_name'] is not None:
        dir_save_path = os.path.join('/mnt', storage_location,'tamirs', 'pressure_field_orientation_dataset', args['folder_name'], f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    elif args["saved_data_dir_path"].startswith('rel_units') or args['saved_data_dir_path'].startswith('rir_shear') or args['saved_data_dir_path'].startswith('rir_uniform'):
        # TO CHANGE: Uncomment this:
        dir_save_path = os.path.join('/mnt', storage_location,'tamirs', 'pressure_field_orientation_dataset', 'robustness_test', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    else:
        # TO CHANGE: Uncomment this:
        # dir_save_path = os.path.join('/mnt', storage_location,'tamirs', 'pressure_field_orientation_dataset', 'big_dataset', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
        if args['save_in_mega']:
            dir_save_path = os.path.join('/mnt', storage_location,'tamirs', 'pressure_field_orientation_dataset', 'mega_dataset', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
        else:
            # dir_save_path = os.path.join('/mnt', storage_location,'tamirs', 'pressure_field_orientation_dataset', f"{args['number_of_angles']}_dataset", f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
            dir_save_path = os.path.join('/mnt', storage_location,'tamirs', 'pressure_field_orientation_dataset', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')


    #dir_save_path = f'indoor_recordings_{num_rotors}_rotors_{num_mics}_mics_d_{delta}_mod_{args["modulate_phase"]}'
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)


    # load phases
    #phase_modulation_path = os.path.join('/mnt', 'walkure_public', 'tamirs', 'phase_modulations', args['phase_modulation_path'])
    #phase_modulation = jnp.load(phase_modulation_path)


    # read params from tb
    num_sources, radiuses_circles, opt_harmonies, _, _, _, _, _, encoder_readings, _,\
        fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)

    # compute position of the rotor overtime
    #cut = min(phase_modulation.shape[-1], encoder_readings.shape[-1])
    #encoder_readings = encoder_readings[:num_rotors, :cut]
    #rotor_position = encoder_readings+phase_modulation
    #onp.save(os.path.join(dir_save_path, 'rotor_position.npy'),rotor_position)



    opt_params, radiuses_circles = read_optimized_params_onp(exp_name, omega, fs, signals_duration, args['num_sources_power'], radiuses_circles, opt_harmonies, args['use_newton_cluster'])

    

    #read_dir = os.path.join('/mnt', 'ssd','tomh', 'rir_data', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #read_dir = os.path.join(cur_dir,'..','..','data','rir', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    #read_dir = os.path.join('/mnt', 'walkure_public','tamirs','rir2d_circular_boundary', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}')
    # read_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'robustness_test', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
    if args['use_small_dense_room']:
        read_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'small_dense_room', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
    elif args['cluster_around_point']:
        if args['use_newton_cluster']:
            read_dir = os.path.join('/', 'home', 'gabrieles', 'EARS', 'data', 'pressure_field_orientation_dataset', 'cluster_point', f"{args['cluster_center'][0]}_{args['cluster_center'][1]}", f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
        else:
            read_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'cluster_point', f"{args['cluster_center'][0]}_{args['cluster_center'][1]}", f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
    elif len(args['corners']) > 0:
        folder_name = 'non_convex_room' if args['folder_name'] is None else args['folder_name']
        if args['use_newton_cluster']:
            read_dir = os.path.join('/', 'home', 'gabrieles', 'EARS', 'data', 'pressure_field_orientation_dataset', folder_name, f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
        else:
            read_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', folder_name, f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
        print(f"RIRs will be loaded from {read_dir}")
    elif args['use_newton_cluster']:
        user = args['user'] # getuser()
        #folder_name = f"{args['number_of_angles']}_angles" if args['number_of_angles'] != 64 else '32_angles'
        if args['folder_name'] is not None:
            folder_name = args['folder_name']
        elif args['number_of_angles'] != 64:
            folder_name = f"{args['number_of_angles']}_angles"
        else:
            folder_name = '32_angles'
        read_dir = os.path.join('/', 'home', user, 'EARS', 'data', 'pressure_field_orientation_dataset', folder_name, f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
    elif args['folder_name'] is not None:
        read_dir = os.path.join('/mnt', storage_location,'tamirs', 'pressure_field_orientation_dataset', args['folder_name'], f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
    elif args["saved_data_dir_path"].startswith('rel_units') or args["saved_data_dir_path"].startswith('rir_shear') or args['saved_data_dir_path'].startswith('rir_uniform'):
        # TO CHANGE: Uncomment this:
        read_dir = os.path.join('/mnt', storage_location,'tamirs', 'pressure_field_orientation_dataset', 'robustness_test',f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
    else:
        # TO CHANGE: Uncomment this:
        # read_dir = os.path.join('/mnt', storage_location,'tamirs', 'pressure_field_orientation_dataset', 'big_dataset',f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
        if args['save_in_mega']:
            read_dir = os.path.join('/mnt', storage_location,'tamirs', 'pressure_field_orientation_dataset', 'mega_dataset',f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
        else:
            # read_dir = os.path.join('/mnt', storage_location,'tamirs', 'pressure_field_orientation_dataset', f"{args['number_of_angles']}_dataset", f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
            read_dir = os.path.join('/mnt', storage_location,'tamirs', 'pressure_field_orientation_dataset', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')


    #read_dir = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/default_5.0_5.0_order_1_0.5_d_0.05/rir"

    rir_dir = os.path.join(read_dir, 'rir_'+args["indoor_mode"])
    extension = ".npy" if not args['compress_data'] else ".hdf5"
    # s = list(filter(lambda x: x.endswith(extension),os.listdir(rir_dir)))
    s = list(filter(lambda x: x.endswith(extension),get_listed_files(rir_dir)))

    # filter s by checking if it already has a corresponding sound
    # already_processed = set(os.listdir(dir_save_path))
    already_processed = set(get_listed_files(dir_save_path))
    not_processed = lambda x: x not in already_processed
    s = list(filter(not_processed, s))
    print(f"Already processed {len(already_processed)}")
    print(f"To compute: {len(s)}")

    # to_cancel = list(filter(lambda x: x.endswith(extension) and x in already_processed, os.listdir(rir_dir)))
    # print(f"{len(to_cancel)} RIRs have already been used to compute the sound but they have not been erased yet. Cancelling now...")
    # for i in tqdm(to_cancel):
    #     os.remove(os.path.join(rir_dir, i))
    # print("Finished to remove any file that had to be removed")
    # exit()
    
    # TO UNCOMMENT
    rirs_paths = sorted(s, key=lambda single_rir: (int(single_rir.strip().split('.')[0].split('_')[0]), int(single_rir.strip().split('.')[0].split('_')[1]), int(single_rir.strip().split('.')[0].split('_')[2])))
    
    # UNCOMMENT to process only part of the data
    # NUM_PARTS = 6
    # PART = 5
    # rirs_paths = rirs_paths[len(rirs_paths)//NUM_PARTS * PART:]
    # print(f"PART: {PART}")
    # print(f"Processing from {len(rirs_paths)//NUM_PARTS * PART} until {len(rirs_paths)}")

    # divide rirs_path into total_gpus_available chunks and process only the index 
    assert args["chunk"] is not None and args["total_gpus_available"] is not None \
        and 0<=args["chunk"]<=args["total_gpus_available"], \
            f"Got args['chunk'] {args['chunk']} and args['total_gpus_available'] {args['total_gpus_available']}, please set them properly"
    start_index = int((len(rirs_paths)/args["total_gpus_available"])*args["chunk"])
    end_index = len(rirs_paths) if (args["chunk"]==(args["total_gpus_available"]-1)) else int((len(rirs_paths)/args["total_gpus_available"])*(args["chunk"]+1))
    rirs_paths = rirs_paths[start_index:end_index]

    # 536 => (133, 253) # paper - random
    # 1984 => (248, 248) # center
    # 0 => (93, 93) # top-left corner
    # 1953 => (248, 93) # center-left
    # 130 => (338, 308) # bottom-right quarter

    print(f'Simulating recordings by exp: {exp_name}')
    number_of_rirs = len(rirs_paths)
    print(f"{number_of_rirs} RIRs to process")
    processed_with_success = 0
    failed_rirs = []
    failed_to_rename = []
    # for idx, rir_path in tqdm(enumerate(rirs_paths), total=len(rirs_paths)):
    for idx, rir_path in enumerate(rirs_paths):

        if idx % 100 == 0:
            print(f"Processing {idx}")

        rir = None
        '''
        for order in range(1, args['max_order']+1):
            rir_dir = os.path.join(read_dir, f'rir_images{order}')
            if order == 1:
                rir = onp.load(os.path.join(rir_dir, rir_path))
            else:
                rir += onp.load(os.path.join(rir_dir, rir_path))
        '''
        
        try:
            if args['compress_data']:
                rir = hdf5.load_numpy(os.path.join(rir_dir, rir_path))
            else:
                rir = onp.load(os.path.join(rir_dir, rir_path))
            processed_with_success += 1
        except Exception as e:
            print(f"Failed to load {rir_path}")
            print(f"got exception {e}")
            failed_rirs.append(rir_path)
            original_name = os.path.join(rir_dir, rir_path)
            corrupted_name = original_name.replace('.npy', '.corrupted', 1) if not args['compress_data'] else original_name.replace('.hdf5', '.corrupted', 1)
            try:
                os.rename(original_name, corrupted_name)
            except Exception as exception:
                print(f"!Failed to rename {original_name} to {corrupted_name}")
                print("Got exception:")
                print(exception)
                failed_to_rename.append(rir_path)
            continue

        origin_x, origin_y, angle = [int(x) for x in rir_path.strip().split('.')[0].split('_')]
        #print(f"Processing {origin_x} {origin_y} {angle}")

        #print(origin_x, origin_y)

        max_len_rir = rir.shape[2]
        max_signal_len = int(fs * signals_duration)
        #max_rec_len = max_len_rir + max_signal_len
        #if max_rec_len % 2 == 1:
            #max_rec_len += 1
        max_rec_len = max_signal_len-max_len_rir+1
        #print(f"Max signal len {max_signal_len} max len rir {max_len_rir} max rec len {max_rec_len}")

        delays_rec = 0 #delays_rec = int(jnp.floor(delays * fs))
        
        phase_shift = jnp.array(args['phase_shift'])
        # simulate recordings based on fitted params
        simulated_recordings = forward_func(opt_params, rir, delays_rec, num_mics, int(max_rec_len),
                                omega, phies0, real_recordings,
                                num_sources, fs, signals_duration, jnp.asarray(opt_harmonies), phase_shift, args['flip_rotation_direction'],
                                radiuses_circles, compare_real=False, num_rotors=num_rotors, modulate_phase=args['modulate_phase'], 
                                recordings_foreach_rotor=True,
                                phase_modulation_injected=None, use_float32=args['compress_data'])
        # if idx == 0:
        #     first_simulated_recordings = simulated_recordings.copy()
        #     first_rir = rir.copy()
        # else:
        #     second_simulated_recordings = simulated_recordings.copy()
        #     second_rir = rir.copy()
        
        # save simulated data. shape: (#rotors, #mics, #sig_len)

        # TO UNCOMMENT!
        if args['compress_data']:
            file_save_path = os.path.join(dir_save_path, f'{origin_x}_{origin_y}_{angle}.hdf5')
            hdf5.save(file_save_path, simulated_recordings)
        else:
            file_save_path = os.path.join(dir_save_path, f'{origin_x}_{origin_y}_{angle}.npy')
            onp.save(file_save_path, simulated_recordings)

        os.remove(os.path.join(rir_dir, rir_path))
    
    print(f"These RIRs have failed:")
    print(failed_rirs)
    print(f"These RIRs have failed to rename:")
    print(failed_to_rename)
    print(f"Processed with success {processed_with_success} RIRs over a total of {number_of_rirs} (failed: {number_of_rirs-processed_with_success}")
    # breakpoint()
        

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
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    # compute_rir_foreach_location(args, delta=0.05, cpu_pool=40)

    '''
    if args['compute_rir_orientation']:
        compute_rir_foreach_location_and_angle(args, delta=args['delta'], delta_angle=args['delta_angle'], cpu_pool=None)
    else:
        compute_rir_foreach_location(args, delta=args['delta'], margins=args['margins'], cpu_pool=40,
                                 num_points_per_side=args['num_points_per_side'],
                                 rotating_angle=args['rotating_angle'])
    '''

    '''
    if args['direct_component']:
        print('Computing only the direct component')
        compute_direct_component()
    else:
        print('Computing only the first order component')
        compute_first_order_image(args)
    '''

    if not args['compute_rir_only'] and not args['compute_sound_only']:
        raise ValueError("Please specify whether to compute RIR (by setting the flag -compute_rir_only) or sound only (by setting the flag -compute_sound_only))")
    if args['compute_rir_only']:
        if args['use_newton_cluster']:
            cpu_pool = int(os.cpu_count())
        elif args['saved_data_dir_path'].startswith('rir_shear'):
            # TO CHANGE
            cpu_pool = 20 #int(os.cpu_count())
        elif args['saved_data_dir_path'].startswith('rir_uniform'):
            cpu_pool = 2
        elif args['saved_data_dir_path'].startswith('rel_units_rir_aspect_ratio') or "aspect_ratio" in args['saved_data_dir_path']:
            cpu_pool = 2
        elif args['saved_data_dir_path'].startswith('rel_units_rir_shear_new_deformation_') or "shear" in args['saved_data_dir_path']:
            cpu_pool = 2
        elif args['saved_data_dir_path'].startswith('rel_units_rir_indoor_absorption_coefficient') or "absorption_coefficient" in args['saved_data_dir_path']:
            cpu_pool = 5
        elif args['saved_data_dir_path'].startswith('rel_units_rir_uniform_new_deformation') or "uniform" in args['saved_data_dir_path']:
            cpu_pool = 3
        elif 'non_convex' in args['saved_data_dir_path']:
            cpu_pool = int(os.cpu_count()*4/5)
        elif args['folder_name'] == "shifted_room":
            cpu_pool = int(os.cpu_count()*2/3)
        elif args['saved_data_dir_path'].startswith("new_shear"):
            cpu_pool = 9
        else:
            # TO CHANGE
            cpu_pool = os.cpu_count() #int(os.cpu_count()*4/5)

        compute_rir_foreach_location_and_angle(args, delta=args['delta'], 
            number_of_angles=args['number_of_angles'], cpu_pool=cpu_pool,
            num_points_per_side=args['num_points_per_side'],
            rotating_angle=args['rotating_angle'],
            margins=args['margins']
            )
    if args['compute_sound_only']:
        simulate_signals_with_rir(args,delta=args['delta'])

    #simulate_signals_with_rir(args,delta=0.05)
    #simulate_by_demand(args)

    
    #compute_rir_foreach_location(args, delta=args['delta'], margins=args['margins'], cpu_pool=2,
                                 #num_points_per_side=args['num_points_per_side'],
                                 #rotating_angle=args['rotating_angle'])
    
    
