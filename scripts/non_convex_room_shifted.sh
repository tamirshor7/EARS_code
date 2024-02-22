#!/bin/bash

python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
 -exp_name alt_Dec20_13-33-20_plato1_final_rads_80mics_128sources_051 \
 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 \
 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 \
 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 4 \
 -num_phases_mics 8 -saved_data_dir_path shifted_non_convex \
    -phase_shift 0 0 0 0 -e_absorption 0.5 \
    -num_points_per_side 50 \
    -flip_rotation_direction 1 0 0 1 -gpu -1 \
    -number_of_angles 64 -compute_rir_only \
    -gpu -1 -fix_corrupted \
    -compress_data \
    -corners 0 0 4 0 4 4 6 4 6 2 9 2 9 7 0 7 \
    -non_convex_offset 0.2 \
    -extra_distance_from_wall 0.17

    #--folder_name "asymmetric_non_convex_room" \
    #-corners 0 0 4 1 5 4 6 3 6 2 9 3 9 5 2 9 2 4 
# create sentinel file
#touch production_complete.txt