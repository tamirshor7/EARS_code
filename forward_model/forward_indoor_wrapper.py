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


from EARS.forward_model.non_convex_room import get_points_in_polygon_from_corners

from EARS.io.fast_io import get_listed_files


def process_location_and_angle(data_in):
    xyt, origin2D, room2D, delays, num_mics, radiuses_circles, num_sources, fs, mics_R, rir_dirs, compress_data_flag = data_in
    x = xyt[0]
    y = xyt[1]
    t = xyt[2]
    origin_sources = origin2D + onp.array([[x, y]])
    origin_mics = [onp.array([x,y])]

    print(f'Processing position: {origin_mics[0]} with angle {t}')
    origin_mics_str = f'{int(x*10**8)}_{int(y*10**8)}_{int(t*10**8)}'
    start_time = time.time()

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
        print(sys.exc_info()[2])


def convert_coordinates(coordinates,n_samples, shear_angle=None, mics_R=None, margins=None, room_x=None,
                        min_coordinate_x=0.93, min_coordinate_y=0.93, max_coordinate_x=4.07, max_coordinate_y=4.07):
    assert len(coordinates.shape)==2 and coordinates.shape[1]==3, f"Coordinates must have shape (n,3). Got {coordinates}"
    assert n_samples<coordinates.shape[0], f"Too many samples asked. Asked {n_samples} from a pool of {coordinates.shape} samples"
    
    if shear_angle is not None:
        # get shearable coordinates
        assert mics_R is not None and margins is not None and room_x is not None, f"mics_R, margins and room_x can't be None but got {mics_R, margins,room_x}"
        shear_angle = onp.deg2rad(shear_angle)
        cotan = 1/onp.tan(shear_angle)
        
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

def compute_xxyytt(args, mics_R, margins, delta, number_of_angles, num_points_per_side, rotating_angle=None):
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

            min_coordinate_x = round(cotan_theta*onp.sqrt(1+1/cotan_theta**2)*(mics_R+margins), 2)
            min_coordinate_y = round((mics_R+margins), 2)

            max_coordinate_x = round(args['room_x'] - cotan_theta*onp.sqrt(1+1/cotan_theta**2)*(mics_R+margins), 2)
            max_coordinate_y = round((args['room_y']-mics_R-margins),2)
            
            xx = onp.linspace(min_coordinate_x, max_coordinate_x, num_points_per_side)
            yy = onp.linspace(min_coordinate_y, max_coordinate_y, num_points_per_side)

            xxyy = onp.array(onp.meshgrid(xx,yy)).T.reshape(-1,2)
            xxyy = shear(xxyy, rotating_angle)
            xxyy = onp.round(xxyy,2)

            xxyy = onp.repeat(xxyy, number_of_angles, 0)

            if number_of_angles < 32:
                seed: int = 0
                tt = onp.random.default_rng(seed).uniform(0, 2*onp.pi, size=((num_points_per_side**2)*number_of_angles))
            else:
                tt = onp.linspace(0, 2*onp.pi, number_of_angles, endpoint=False)
                tt = onp.tile(tt, (num_points_per_side**2))

            xxyytt = onp.concatenate((xxyy, onp.expand_dims(tt, axis=-1)), axis=-1).reshape(-1,3)

        else:
            xx = onp.linspace( round((mics_R+margins), 2), round((args['room_x']-mics_R-margins),2), num_points_per_side)
            yy = onp.linspace( round((mics_R+margins), 2), round((args['room_y']-mics_R-margins),2), num_points_per_side)

            if args['use_newton_cluster']:
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
    return xxyytt

def define_save_dir(args, delta, number_of_angles):
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
        user = args['user']
        if args['folder_name'] is not None:
            folder_name = args['folder_name']
        elif number_of_angles != 64:
            folder_name = f'{number_of_angles}_angles'
        else:
            folder_name = '32_angles'
        save_dir = os.path.join('/', 'home', user, 'EARS', 'data', 'pressure_field_orientation_dataset', folder_name, f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
    elif args['folder_name'] is not None:
        save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', args['folder_name'], f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')         
    elif args['saved_data_dir_path'].startswith('rel_units') or args['saved_data_dir_path'].startswith('consistent') or args['saved_data_dir_path'].startswith('rir_shear') or args['saved_data_dir_path'].startswith('rir_uniform'):
        save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'robustness_test', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')     
    else:
        if args['save_in_mega']:
            save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', 'mega_dataset', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
        else:
            #save_dir = os.path.join('/mnt', 'walkure_public','tamirs', 'pressure_field_orientation_dataset', f'{args["saved_data_dir_path"]}_{args["room_x"]}_{args["room_y"]}_order_{args["max_order"]}_{round(args["e_absorption"],2)}_d_{delta}', 'rir')
            save_dir = os.path.join(args['saved_data_dir_path'], 'rir')
    return save_dir

def compute_rir_foreach_location_and_angle(args, delta=0.02, delta_angle=onp.pi/8, margins=0.02, cpu_pool=40, by_demand=False,
                                           number_of_angles=16, num_points_per_side=None, rotating_angle=None):
    print('Building the room')
    num_mics = args['num_phases_mics']
    num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter,\
            fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)
    mics_R = 2*rotor_length_meter
    origin2D -= onp.array([[args['org_x'], args['org_y']]])

    # Compute xxyytt based on the provided arguments
    xxyytt = compute_xxyytt(args, mics_R, margins, delta, number_of_angles, num_points_per_side, rotating_angle)
    
    # Define save directory based on the provided arguments
    save_dir = define_save_dir(args, delta, number_of_angles)

    os.makedirs(save_dir, exist_ok=True)
    rir_indoor_dir = os.path.join(save_dir, 'rir_indoor')
    

    rir_dirs = [rir_indoor_dir]
    for rir_dir_path in rir_dirs:
        if not os.path.exists(rir_dir_path):
            os.makedirs(rir_dir_path)
    
    process_x = lambda x: (
        int(x.strip().split('.')[0].split('_')[0]), 
        int(x.strip().split('.')[0].split('_')[1]), 
        int(x.strip().split('.')[0].split('_')[2])
    )
    
    computed_files = set(filter(lambda x: '.' in x and not x.endswith('.txt'), get_listed_files(rir_dirs[-1])))


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

    sound_folder: str = get_listed_files(os.path.join(rir_dirs[-1], '..', '..'))


    computed_sound = set(filter(lambda x:  '.' in x and not x.endswith('.txt'), sound_folder))
    
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

    number_filter = lambda x: x.strip().split('.')[0].split('_')[0].isdigit() and x.strip().split('.')[0].split('_')[1].isdigit() \
                              and x.strip().split('.')[0].split('_')[2].isdigit()
    computed_files = set(filter(number_filter, computed_files))
    already_processed = set([process_x(x) for x in computed_files])


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


def simulate_signals_with_rir(args):
    num_mics = args['num_phases_mics']
    num_rotors = args['num_rotors']
    
    dir_save_path = define_save_dir(args, args['delta'], number_of_angles=args['number_of_angles'])
    dir_save_path = os.path.join(dir_save_path, '..')

    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)

    # read params from tb
    num_sources, radiuses_circles, opt_harmonies, _, _, _, _, _, encoder_readings, _,\
        fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)

    opt_params, radiuses_circles = read_optimized_params_onp(exp_name, omega, fs, signals_duration, args['num_sources_power'], radiuses_circles, opt_harmonies, args['use_newton_cluster'])

    read_dir = os.path.join(dir_save_path, 'rir')

    rir_dir = os.path.join(read_dir, 'rir_'+args["indoor_mode"])
    extension = ".npy" if not args['compress_data'] else ".hdf5"
    s = list(filter(lambda x: x.endswith(extension),get_listed_files(rir_dir)))

    # filter s by checking if it already has a corresponding sound
    already_processed = set(get_listed_files(dir_save_path))
    not_processed = lambda x: x not in already_processed
    s = list(filter(not_processed, s))
    print(f"Already processed {len(already_processed)}")
    print(f"To compute: {len(s)}")

    rirs_paths = sorted(s, key=lambda single_rir: (int(single_rir.strip().split('.')[0].split('_')[0]), int(single_rir.strip().split('.')[0].split('_')[1]), int(single_rir.strip().split('.')[0].split('_')[2])))

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
    for idx, rir_path in enumerate(rirs_paths):

        if idx % 100 == 0:
            print(f"Processing {idx}")

        rir = None
        
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
        
        

        max_len_rir = rir.shape[2]
        max_signal_len = int(fs * signals_duration)
        max_rec_len = max_signal_len-max_len_rir+1

        delays_rec = 0 
        
        phase_shift = jnp.array(args['phase_shift'])
        # simulate recordings based on fitted params
        simulated_recordings = forward_func(opt_params, rir, delays_rec, num_mics, int(max_rec_len),
                                omega, phies0, real_recordings,
                                num_sources, fs, signals_duration, jnp.asarray(opt_harmonies), phase_shift, args['flip_rotation_direction'],
                                radiuses_circles, compare_real=False, num_rotors=num_rotors, modulate_phase=args['modulate_phase'], 
                                recordings_foreach_rotor=True,
                                phase_modulation_injected=None, use_float32=args['compress_data'])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

    if not args['compute_rir_only'] and not args['compute_sound_only']:
        args['compute_rir_only'] = True
        args['compute_sound_only'] = True

    if args['compute_rir_only']:
        cpu_pool = os.cpu_count()
        compute_rir_foreach_location_and_angle(args, delta=args['delta'], 
            number_of_angles=args['number_of_angles'], cpu_pool=cpu_pool,
            num_points_per_side=args['num_points_per_side'],
            rotating_angle=args['rotating_angle'],
            margins=args['margins']
            )
    if args['compute_sound_only']:
        simulate_signals_with_rir(args)
