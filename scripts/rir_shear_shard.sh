#!/bin/bash


compute_sound() {
    shear_angle="$1"
    which_gpu="$2"
    chunk="$3"
    total_number_of_chunks="$4"

    echo "Computing sound with shear angle $shear_angle on gpu $which_gpu"
    echo "Running on chunk $chunk/$total_number_of_chunks on $which_gpu"

    python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
        -exp_name alt_Dec20_13-33-20_plato1_final_rads_80mics_128sources_051 \
        -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
      -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
      -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "new_shear/s_$shear_angle" \
      -phase_shift 0 0 0 0 -e_absorption 0.5 -delta 0.05 -num_points_per_side 25 \
      -room_x 5.0 -room_y 5.0 -margins 0.02 -rotating_angle "$shear_angle" \
      -flip_rotation_direction 1 0 0 1 -number_of_angles 64 \
        -gpu "$which_gpu" -compute_sound_only -chunk "$chunk" -total_gpus_available "$total_number_of_chunks" \
        -compress_data
    
    echo "Absorption coefficient $shear_angle done"
}

if [ "$0" = "$BASH_SOURCE" ]; then
    # Check if at least one parameter is provided
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <shear angle> <number of chunks> <total number of gpus>"
        exit 1
    fi

    shear_angle=$1
    number_of_chunks=$2
    total_number_of_available_gpus=$3
    gpu_index=0
    # Iterate over the values and run the function concurrently
    for chunk in $(seq 0 1 $number_of_chunks); do        
        compute_sound "$shear_angle" "$gpu_index" "$chunk" "$total_number_of_available_gpus" &
        gpu_index=$(((gpu_index+1)%total_number_of_available_gpus))
    done

    # Wait for all background processes to complete
    wait

    echo "All functions have completed"
    
fi