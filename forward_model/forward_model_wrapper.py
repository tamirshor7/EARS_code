import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)
import numpy as onp
import os
import pdb
import argparse
from time import time
from itertools import product
#import h5py
from multiprocessing import Pool
from forward_model import add_args, build_room_and_read_gt_data, read_optimized_params_onp, forward_func
from pyroomacoustics_differential import forward_model_2D_interface, jax_forward_model_2D_interface

from EARS.io import hdf5, fast_io
from re import compile, search

# python forward_model_wrapper.py -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 2 -max_order 2 -absorption 0.2 -org_x 2.8 -org_y 3.1

# origins = [ (1.2, 1.7), (1.4, 3.8), (2.5, 2.5), (2.5, 3.5), (3.6, 3.7), 
#             (1.9, 2.1), (2.8, 3.1), (3.2, 2.1), (2.4, 1.7)]
# python forward_model_wrapper.py -gpu 1 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -max_order 1 -num_rotors 2 -saved_data_dir_path rir_all_2.8_3.1_phase_mod -org_x 2.8 -org_y 3.1 -phase_shift 0 0 -modulate_phase

# simulate 4 rotors:
# python forward_model_wrapper.py -gpu 3 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 4 -phase_shift 0 0 0 0 -max_order 7 -e_absorption 0.2 -org_x 2.5 -org_y 2.5 -saved_data_dir_path entire_room_4_rotors_2.5_2.5

def process_location(data_in):
    yi, origin2D, room2D, delays, mics_distances, phi_mics, radiuses_circles, num_sources, fs, max_order, delta_mics, rir_dirs, drone_orientation = data_in
    yi_2 = round(yi, 2)

    print(f'Processing {yi_2}')
    # use forward_model_2D_interface to use numpy
    # rir: RIR of the maximum order
    # rir_by_orders: RIR of each order separately up to max_order (it includes the data contained in rir)
    _, _, rir, rir_by_orders = forward_model_2D_interface.compute_ISM_and_RIR_phased_sources_array(
                                    origin2D, origin2D,
                                    room2D, delays,
                                    mics_distances, phi_mics=phi_mics, 
                                    coords_mics=None, num_mics_circ_array=0, radiuses_circles=radiuses_circles, num_sources_in_circle=num_sources,
                                    sources_distances=None, phi_sources_array=None, phi0_sources=drone_orientation,
                                    fs=fs, max_order=max_order,
                                    grid_mics_y=yi_2, delta_mics=delta_mics, mics_R=max(radiuses_circles), 
                                    enable_prints=True, is_plot_room=False, room_plot_out='mics_plotted.png')
    print(f'Rir of {yi_2} computed')
    num_mics = len(rir)
    max_len_rir = max([len(rir[i][j])
                        for i, j in product(range(num_mics), range(len(rir[0])))])
    
    # pad rir with zeros to make a numpy - will be used late on for convolving
    # and save npy to dirs
    rir = onp.array([[onp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir[j]] for j in range(len(rir))])
    # onp.save(os.path.join(rir_dirs[0], str(yi_2)+'.npy'), rir)
    hdf5.save(os.path.join(rir_dirs[0], str(yi_2)+'.hdf5'), rir)

    for k in range(max_order+1):
        rir_by_order = onp.array([[onp.pad(i, (0, int(max_len_rir)-len(i))) for i in rir_by_orders[k][j]] for j in range(len(rir_by_orders[k]))])
        # onp.save(os.path.join(rir_dirs[k+1], str(yi_2)+'.npy'), rir_by_order)
        hdf5.save(os.path.join(rir_dirs[k+1], str(yi_2)+'.hdf5'), rir_by_order)

def compute_rir_only(source_path, y, 
                    args, origin2D, room2D, delays, mics_distances, phi_mics, radiuses_circles, num_sources, fs, cpu_pool=40):

    rir_indoor_dir = os.path.join(source_path, 'rir_indoor')
    rir_dirs = [rir_indoor_dir] + [os.path.join(source_path, 'rir_images'+str(k)) for k in range(args['max_order']+1)]

    for rir_dir_path in rir_dirs:
        if not os.path.exists(rir_dir_path):
            os.makedirs(rir_dir_path)

    # filter out the files that have been already computed with the right shape
    computed_files = []
    for rir_dir_path in rir_dirs:
        computed_files.extend(
            [os.path.join(rir_dir_path, x) for x in fast_io.get_listed_files(rir_dir_path)]
            )
    good_files = set(
            map(lambda x: os.path.basename(x),
                filter(lambda x: hdf5.get_shape(x)[0] == 250, 
                    computed_files))
        )
    good_y = set(map(lambda x: round(float(x.removesuffix(".hdf5")),2), good_files))
    
    y = list(filter(lambda x: x not in good_y, y.tolist()))

    drone_orientation = args['drone_orientation']
    data_inputs = [[yi, origin2D, room2D, delays, mics_distances, phi_mics, radiuses_circles, num_sources, fs, args['max_order'], args['delta_mics'], rir_dirs, drone_orientation] for yi in y]
    with Pool(cpu_pool) as pool:
         pool.map(process_location, data_inputs)
    #for data_in in data_inputs:
        #breakpoint()
        #process_location(data_in)
         
def convert_rir_to_sound(data):
    rir_dir, rir_path, max_len_rir, max_rec_len, flip_rotation_direction, num_rotors, modulate_phase = data
    cur_rir = hdf5.load_numpy(os.path.join(rir_dir, rir_path))
    
    yi_2 = float(rir_path.strip().removesuffix(".hdf5")) # remove '.npy' and return yi_2
    print(f'Processing y={yi_2}')

    cur_len_rir = cur_rir.shape[2]
    to_pad = max_len_rir-cur_len_rir
    padding_size = ((0,0), (0,0), (0,to_pad))
    cur_rir = onp.pad(cur_rir, padding_size, mode="constant", constant_values=0)
    num_mics = cur_rir.shape[0]
    # max_rec_len = max_len_rir + max_signal_len
    # if max_rec_len % 2 == 1:
    #     max_rec_len += 1
    #max_rec_len = max_signal_len-max_len_rir+1
    delays_rec = 0 #delays_rec = int(jnp.floor(delays * fs))
    start_time = time()
    # simulate recordings based on fitted params
    simulated_recordings = forward_func(opt_params, cur_rir, delays_rec, num_mics, int(max_rec_len),
                            omega, phies0, real_recordings,
                            num_sources, fs, signals_duration, jnp.asarray(opt_harmonies), phase_shift, flip_rotation_direction,
                            radiuses_circles, compare_real=False, num_rotors=num_rotors, modulate_phase=modulate_phase, recordings_foreach_rotor=True,
                            phase_modulation_injected=None, use_float32=True)
    # sim_path = os.path.join(saved_data_dir_path, f'{indoor_mode}_y_{yi_2}.npy')
    try:
        sim_path = os.path.join(saved_data_dir_path, f'{indoor_mode}_y_{yi_2}.hdf5')
        # jnp.save(sim_path, simulated_recordings)
        hdf5.save(sim_path, simulated_recordings)
        os.remove(os.path.join(rir_dir, rir_path))
        print(f"Computed {yi_2} in {time()-start_time}")
    except Exception as e:
        print(f'Error in saving {sim_path} - {e}')
        

def simulate_signals_with_rir(source_path, saved_data_dir_path, 
                                indoor_mode, modulate_phase, num_rotors,
                                fs, signals_duration, opt_params,
                                omega, phies0, real_recordings, num_sources,
                                opt_harmonies, radiuses_circles, phase_shift, flip_rotation_direction,
                                cpu_pool=30):
    rir_dir = os.path.join(source_path, f'rir_{indoor_mode}')
    
    # s = os.listdir(rir_dir)
    s = fast_io.get_listed_files(rir_dir)

    rirs_paths = sorted(s, key=lambda single_rir: (float(single_rir.strip().removesuffix(".hdf5"))))

    # create directory to save data
    if not os.path.exists(saved_data_dir_path):
        os.makedirs(saved_data_dir_path)

    max_signal_len = int(fs * signals_duration)

    # max_len_rir = max(onp.load(os.path.join(rir_dir,rir_path), mmap_mode="r").shape[-1] for rir_path in rirs_paths)
    max_len_rir = max(hdf5.get_shape(os.path.join(rir_dir, rir_path))[-1] for rir_path in rirs_paths)
    max_rec_len = max_signal_len-max_len_rir+1

    data = [[rir_dir, rir_path, max_len_rir, max_rec_len, flip_rotation_direction, num_rotors, args['modulate_phase']] for rir_path in rirs_paths]
    # start_time = time()
    # for rir_path in rirs_paths:
    #     # cur_rir = onp.load(os.path.join(rir_dir, rir_path))
    #     convert_rir_to_sound(data)

    with Pool(cpu_pool) as pool:
        pool.map(convert_rir_to_sound, data)
        
    # print(f'running time: {time()-start_time}')

def relative2absolute_coordinates(coordinates: onp.ndarray):
    return coordinates*(4.07-0.93)+0.93

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu'] #'2, 3'#
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1" #'2, 3'#
    # super important flags for JAX:
    # os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    #os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.2'

    num_sources, radiuses_circles, opt_harmonies, origin2D, room2D, delays, mics_distances, phi_mics, encoder_readings, rotor_length_meter, \
                    fs, signals_duration, omega, phies0, phase_shift, real_recordings, exp_name = build_room_and_read_gt_data(args)


    opt_params, radiuses_circles = read_optimized_params_onp(exp_name, omega, fs, signals_duration, args['num_sources_power'], radiuses_circles, opt_harmonies)

    # create an empty df to save the processed data
    #source_path = os.path.join('/mnt', 'ssd','tomh', 'rir_data',  f'entire_room_org_{args["org_x"]}_{args["org_y"]}_rotors_{args["num_rotors"]}_final1_fine')
    #source_path = os.path.join('/mnt', 'walkure_public','tomh', 'rir',  f'entire_room_org_{args["org_x"]}_{args["org_y"]}_rotors_{args["num_rotors"]}_final1_fine')
    # source_path = os.path.join('/mnt', 'walkure_public', 'tamirs', 'simulator', f"{args['saved_data_dir_path']}")
    
    #entire_room_org_2.5_2.5_rotors_4_free_space_and_indoor
    # source_path = os.path.join('/mnt', 'walkure_public', 'tamirs', 'simulator', f'entire_room_org_{args["org_x"]}_{args["org_y"]}_rotors_{args["num_rotors"]}_free_space_and_indoor')
    source_path = os.path.join('/mnt', 'walkure_public', 'tamirs', 'simulator', 'pressure_field_visualization_rirs', f'entire_room_org_{args["org_x"]}_{args["org_y"]}_rotors_{args["num_rotors"]}_free_space_and_indoor')
    saved_data_dir_path=os.path.join('/mnt', 'walkure_public', 'tamirs', 'simulator', 'raw_data',  f'entire_room_org_{args["org_x"]}_{args["org_y"]}_rotors_{args["num_rotors"]}_free_space_and_indoor')

    #source_path = os.path.join('/home', 'gabrieles', 'ears', 'code', 'data', 'visualization_rir', f"{args['saved_data_dir_path']}")

    #breakpoint()
    y = onp.arange(0, args['room_y'], args['delta_mics'])
    
    cpu_pool = 30
    compute_rir_only(source_path, y, args, origin2D, room2D, delays, mics_distances, phi_mics, radiuses_circles, num_sources, fs, cpu_pool=cpu_pool)
    print('RIRs computed')
    #indoor_modes = [f'images{i}' for i in range(args["max_order"]+1)] + ['indoor']
    indoor_modes = ['images0', 'indoor']
    
    for indoor_mode in indoor_modes:
        print(f"Computing signals for {indoor_mode}")
        # simulate_signals_with_rir(source_path, args['saved_data_dir_path'], 
        simulate_signals_with_rir(source_path, saved_data_dir_path, 
                                    indoor_mode, args['modulate_phase'], args['num_rotors'],
                                    fs, signals_duration, opt_params,
                                    omega, phies0, real_recordings, num_sources,
                                    opt_harmonies, radiuses_circles, phase_shift, args['flip_rotation_direction'],
                                    cpu_pool=cpu_pool)
    

# python forward_model_wrapper.py -gpu 3 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 1 -phase_shift 0 0 0 0 -max_order 3 -e_absorption 0.2 -org_x 2.5 -org_y 2.5 -saved_data_dir_path entire_room_org_2.5_2.5_rotors_1_new
# python forward_model_wrapper.py -gpu 2 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 1 -phase_shift 0 -max_order 6 -e_absorption 0.2 -org_x 2.5 -org_y 3.0 -delta_mics 0.2 -saved_data_dir_path entire_room_org_2.5_3.0_rotors_1_final1_fine_reversed

# python forward_model_wrapper.py -gpu 2 -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 4 -phase_shift 0 0 0 0 -max_order 6 -e_absorption 0.2 -org_x 2.5 -org_y 2.5 -delta_mics 0.02 -saved_data_dir_path entire_room_org_2.5_2.5_rotors_4_final1_fine_0011 -flip_rotation_direction 0 0 1 1