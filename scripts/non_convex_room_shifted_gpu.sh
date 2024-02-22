#!/bin/bash

compute_sound(){
    chunk="$1"
    total_number_of_chunks="$2"
    which_gpu="$3"

    echo "Running on chunk $chunk/$total_number_of_chunks on $which_gpu"

    python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
 -exp_name alt_Dec20_13-33-20_plato1_final_rads_80mics_128sources_051 \
 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 \
 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 \
 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 4 \
 -num_phases_mics 8 -saved_data_dir_path shifted_non_convex \
    -phase_shift 0 0 0 0 -e_absorption 0.5 \
    -num_points_per_side 50 \
    -flip_rotation_direction 1 0 0 1 -gpu "$which_gpu" \
    -number_of_angles 64 -compute_sound_only \
    -fix_corrupted \
    -compress_data \
    -corners 0 0 4 0 4 4 6 4 6 2 9 2 9 7 0 7 \
    -non_convex_offset 0.2 \
    -extra_distance_from_wall 0.17 \
    -chunk "$chunk" -total_gpus_available "$total_number_of_chunks"

    echo "Done running on chunk $chunk on $which_gpu / $total_gpus_available"

}

# Check if at least one parameter is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <number of chunks> <total number of gpus>"
    exit 1
fi

number_of_chunks=$1
total_number_of_available_gpus=$2
gpu_index=-1

# Display the parameter
echo "Number of chunks $number_of_chunks, number of gpus available $total_number_of_available_gpus"

while [ $(ls -A "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/non_convex_room/shifted_non_convex_5.0_5.0_order_1_0.5_d_0.05/rir/rir_indoor/" | wc -l) -ne 0 ]; do
    for current_chunk in $(seq 0 1 $number_of_chunks); do
        gpu_index=$(((gpu_index+1)%total_number_of_available_gpus))
        compute_sound "$current_chunk" "$number_of_chunks" "$gpu_index" &
    done
    wait
    gpu_index=-1
done