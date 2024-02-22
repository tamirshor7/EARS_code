#!/bin/bash

# Define your function with parameters
my_function() {
    param1="$1"
    
    room_x=$(python -c "from math import sqrt; side = sqrt(1+$param1)*5; print(round(side, 2))")
    room_y=$(python -c "from math import sqrt; side = 5/sqrt(1+$param1); print(round(side, 2))")
    delta=0.05
    margins=0.02

    # Your function logic here
    echo "Running function with param1=$param1 room_side=$room_side delta=$delta margins=$margins"
    
    # python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
    #  -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
    # -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
    #  -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
    #  -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path rel_units_rir_aspect_ratio \
    #  -phase_shift 0 0 0 0 -e_absorption 0.5 -delta "$delta" -num_points_per_side 25 \
    #  -room_x "$room_x" -room_y "$room_y" -margins "$margins" \
    #  -flip_rotation_direction 1 0 0 1 -number_of_angles 2 \
    #  -compute_rir_only -gpu -1 -fix_corrupted \
    #  -use_relative_units \
    #   -reference_coordinates_path "/mnt/walkure_public/tamirs/unique_test_coordinates.npy" \
    #   -n_samples 1250 > "$param1_progress.out"

    python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
     -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
    -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
     -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
     -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "aspect_ratio/asp_$param1" \
     -phase_shift 0 0 0 0 -e_absorption 0.5 -delta "$delta" -num_points_per_side 25 \
     -room_x "$room_x" -room_y "$room_y" -margins "$margins" \
     -flip_rotation_direction 1 0 0 1 -number_of_angles 64 \
     -compute_rir_only -gpu -1 -fix_corrupted \
      -compress_data > "$param1_progress.out" 
    
    echo "Function with param1=$param1 room_x=$room_x room_y=$room_y delta=$delta margins=$margin"
}

compute_sound() {
    aspect_ratio="$1"
    which_gpu="$2"

    room_x=$(python -c "from math import sqrt; side = sqrt(1+$param1)*5; print(round(side, 2))")
    room_y=$(python -c "from math import sqrt; side = 5/sqrt(1+$param1); print(round(side, 2))")
    delta=0.05
    margins=0.02

    echo "Computing sound with aspect_ratio $aspect_ratio on gpu $which_gpu"

    # python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
    #     -exp_name alt_Dec20_13-33-20_plato1_final_rads_80mics_128sources_051 \
    #     -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
    #   -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
    #   -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path rel_units_rir_aspect_ratio \
    #   -phase_shift 0 0 0 0 -e_absorption 0.5 -delta 0.05 -num_points_per_side 25 \
    #   -room_x "$room_x" -room_y "$room_y" -margins "$margins" -flip_rotation_direction 1 0 0 1 \
    #   -number_of_angles 2 \
    #     -gpu "$which_gpu" -compute_sound_only -chunk 0 -total_gpus_available 1 \
    #     > aspect_ratio_$aspect_ratio.out 

    python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
        -exp_name alt_Dec20_13-33-20_plato1_final_rads_80mics_128sources_051 \
        -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
      -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
      -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "aspect_ratio/asp_$aspect_ratio" \
      -phase_shift 0 0 0 0 -e_absorption 0.5 -delta 0.05 -num_points_per_side 25 \
      -room_x "$room_x" -room_y "$room_y" -margins "$margins" -flip_rotation_direction 1 0 0 1 \
      -number_of_angles 64 \
        -gpu "$which_gpu" -compute_sound_only -chunk 0 -total_gpus_available 1 \
        -compress_data > aspect_ratio_$aspect_ratio.out 
    
    echo "Aspect ratio $aspect_ratio done"
}


# Check if at least one parameter is provided
# if [ $# -eq 0 ]; then
#     echo "Usage: $0 <total number of gpus>"
#     exit 1
# fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <cpu|gpu> <total number of gpus>"
    exit 1
fi

total_number_of_available_gpus=$2
gpu_index=0


# Define the range of values
# start=0.1
# end=1.0
# step=0.05
start=-0.85
end=0.95
step=0.05

case "$1" in
    "cpu")
        # Iterate over the values and run the function concurrently
        for current_value in $(seq $start $step $end); do
            param1="$current_value"
            
            # Run the function in the background
            my_function "$param1" &
            #compute_sound "$param1" "$gpu_index" &
            #gpu_index=$(((gpu_index+1)%total_number_of_available_gpus))
        done

        # Wait for all background processes to complete
        wait

        echo "All functions have completed"
        ;;
    "gpu")
        # Iterate over the values and run the function concurrently
        for current_value in $(seq $start $step $end); do
            param1="$current_value"
            
            # Run the function in the background
            #my_function "$param1" &
            compute_sound "$param1" "$gpu_index" &
            gpu_index=$(((gpu_index+1)%total_number_of_available_gpus))
        done

        # Wait for all background processes to complete
        wait

        echo "All functions have completed"
        ;;
    *)
        echo "Error: Invalid argument. Please use 'cpu' or 'gpu'"
        exit 1
        ;;
esac


# Iterate over the values and run the function concurrently
# for current_value in $(seq $start $step $end); do
#     param1="$current_value"
    
#     # Run the function in the background
#     #my_function "$param1" &
#     compute_sound "$param1" "$gpu_index" &
#     gpu_index=$(((gpu_index+1)%total_number_of_available_gpus))
# done

# # Wait for all background processes to complete
# wait

# echo "All functions have completed"
# echo "Running evaluation test..."
# export CUDA_VISIBLE_DEVICES=0
# python EARS/localization/evaluate.py --model_path "/mnt/walkure_public/tamirs/EARS/current_best_model/pipe_max_freq_const_90_degs_only_recons_epoch_100/summary/test/best_model.pt" --robustness_test aspect_ratio
# echo "Finished running test"