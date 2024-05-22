#!/bin/bash

generate_rirs() {
    param1="$1"

    echo "Running function with param1=$param1"
    
    python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
     -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
     -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
      -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
      -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "absorption_coefficient/e_$param1" \
      -phase_shift 0 0 0 0 -e_absorption "$param1" -delta 0.05 -num_points_per_side 25 \
      -room_x 5.0 -room_y 5.0 -flip_rotation_direction 1 0 0 1 -number_of_angles 64 \
      -compute_rir_only -gpu -1 -fix_corrupted -compress_data \
      -generation_computation "iterative" > absorption_coefficient_$param1.out 

    echo "Function with param1=$param1"
}

compute_sound() {
    absorption_coefficient="$1"
    which_gpu="$2"

    echo "Computing sound with absorption coefficient $absorption_coefficient on gpu $which_gpu"

    python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
        -exp_name alt_Dec20_13-33-20_plato1_final_rads_80mics_128sources_051 \
        -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
      -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
      -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "absorption_coefficient/e_$absorption_coefficient" \
      -phase_shift 0 0 0 0 -e_absorption "$absorption_coefficient" -delta 0.05 -num_points_per_side 25 \
      -room_x 5.0 -room_y 5.0 -number_of_angles 64 -flip_rotation_direction 1 0 0 1 \
        -gpu "$which_gpu" -compute_sound_only -chunk 0 -total_gpus_available 1 \
        -compress_data > absorption_coefficient_$absorption_coefficient.out 
    
    echo "Absorption coefficient $absorption_coefficient done"
}

# Check if at least one parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <cpu|gpu> <total number of gpus>"
    exit 1
fi

# Define the range of values
start=0.05
end=0.95
step=0.05

case "$1" in
    "cpu")

        for current_value in $(seq $start $step $end); do
            param1="$current_value"
            
            my_function "$param1" & 
        done

        # Wait for all background processes to complete
        wait
        echo "All functions have completed"
        ;;
    "gpu")
        total_number_of_available_gpus=$2
        gpu_index=0

        # Iterate over the values and run the function concurrently
        for current_value in $(seq $start $step $end); do
            param1="$current_value"
            
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