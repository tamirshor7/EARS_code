#!/bin/bash

# Define your function with parameters
my_function() {
    param1="$1"
    #param2="$2"
    
    room_side=$(python -c "from math import sqrt; side = sqrt($param1)*5 if $param1>=1 else (sqrt($param1) +((2.0*(1-sqrt($param1))*0.9144+2.0*0.02)/5.0))*5.0; print(round(side, 6))")
    #room_y=$(echo "5 * $param1" | bc)
    #delta=$(echo "sqrt(($param1*9.80441344)/3963)" | python -c "print(round(float(raw_input()),2))) 
    # param1 is the deformation
    # Formula for delta: delta = sqrt(param1*old_area/number_of_datapoints)
    #delta=$(python -c "from math import sqrt; delta = sqrt($param1*9.80441344/3963); print(round(delta, 6))")
    # Formula for margins: margins = 2*sqrt(param1)*old_margin + 2*(sqrt(param1)-1)*mics_R
    margins=$(python -c "from math import sqrt; new_margin = (sqrt($param1)*0.02 + (sqrt($param1)-1)*0.9144) if $param1>=1 else 0.02; print(round(new_margin, 6)) ")

    # Your function logic here
    echo "Running function with param1=$param1 room_side=$room_side margins=$margins"
    
    # python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
    #  -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
    # -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
    #  -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
    #  -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path rel_units_rir_uniform_new_deformation \
    #  -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 -delta 0.05 \
    #  -room_x "$room_side" -room_y "$room_side" -margins "$margins" \
    #  -flip_rotation_direction 1 0 0 1 -number_of_angles 2 \
    #  -compute_rir_only -gpu -1 -fix_corrupted \
    #  -use_relative_units \
    #  -reference_coordinates_path "/mnt/walkure_public/tamirs/unique_test_coordinates.npy"\
    #  -n_samples 1250 > progress_uniform_$param1.out 

    python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
     -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
    -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
     -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
     -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "uniform/u_$param1" \
     -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 -delta 0.05 \
     -room_x "$room_side" -room_y "$room_side" -margins "$margins" \
     -flip_rotation_direction 1 0 0 1 -number_of_angles 64 \
     -compute_rir_only -gpu -1 -fix_corrupted \
      -compress_data > progress_uniform_$param1.out 
    
    echo "Function with param1=$param1 room_side=$room_side delta=$delta margins=$margin"
}

compute_sound() {
    param1="$1"
    which_gpu="$2"

    room_side=$(python -c "from math import sqrt; side = sqrt($param1)*5 if $param1>=1 else (sqrt($param1) +((2.0*(1-sqrt($param1))*0.9144+2.0*0.02)/5.0))*5.0; print(round(side, 6))")
    margins=$(python -c "from math import sqrt; new_margin = (sqrt($param1)*0.02 + (sqrt($param1)-1)*0.9144) if $param1>=1 else 0.02; print(round(new_margin, 6)) ")


    echo "Computing sound with factor $param1 on gpu $which_gpu"

    # python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
    #     -exp_name alt_Dec20_13-33-20_plato1_final_rads_80mics_128sources_051 \
    #     -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
    #   -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
    #   -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path rel_units_rir_uniform_new_deformation \
    #   -phase_shift 0 0 0 0 -e_absorption 0.5 -delta 0.05 -num_points_per_side 25 \
    #   -room_x "$room_side" -room_y "$room_side" -margins "$margins" \
    #   -flip_rotation_direction 1 0 0 1 -number_of_angles 2 \
    #     -gpu "$which_gpu" -compute_sound_only -chunk 0 -total_gpus_available 1 \
    #     > progress_uniform_$factor.out

    python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
        -exp_name alt_Dec20_13-33-20_plato1_final_rads_80mics_128sources_051 \
        -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
      -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
      -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "uniform/u_$param1" \
      -phase_shift 0 0 0 0 -e_absorption 0.5 -delta 0.05 -num_points_per_side 25 \
      -room_x "$room_side" -room_y "$room_side" -margins "$margins" \
      -flip_rotation_direction 1 0 0 1 -number_of_angles 64 \
        -gpu "$which_gpu" -compute_sound_only -chunk 0 -total_gpus_available 1 \
        -compress_data > progress_uniform_$param1.out 
    
    echo "Absorption coefficient $param1 done"
}

# Define an array of parameter sets
#params=(
#    "param1_value1 param2_value1"
#    "param1_value2 param2_value2"
#    "param1_value3 param2_value3"
#)

# Loop through the parameter sets and run the function concurrently
#for param_set in "${params[@]}"; do
    # Split the parameter set into individual parameters
    #param1=$(echo "$param_set" | awk '{print $1}')
    #param2=$(echo "$param_set" | awk '{print $2}')
    
    # Run the function in the background
    #my_function "$param1" "$param2" &
#done

#for ((i = 0; i < ${#param1_values[@]}; i++)); do
    #param1="${param1_values[i]}"
    #param2="${param2_values[i]}"
    
    # Run the function in the background
    #my_function "$param1" "$param2" &
#done

if [ "$0" = "$BASH_SOURCE" ]; then
    # Check if at least one parameter is provided
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <cpu|gpu> <total number of gpus>"
        exit 1
    fi

    total_number_of_available_gpus=$2
    gpu_index=0

  # Define the range of values
    start=0.5
    end=2.0
    step=0.05

    case "$1" in
        "cpu")
            # Iterate over the values and run the function concurrently
        for current_value in $(seq $start $step $end); do
            param1="$current_value"
            #param2="some_value"  # You can set param2 to a constant value
            
            # Run the function in the background
            my_function "$param1" & #"$param2" &
            
        done

        # Wait for all background processes to complete
        wait

        echo "All functions have completed"
            ;;
        "gpu")
            # Iterate over the values and run the function concurrently
            for current_value in $(seq $start $step $end); do
                param1="$current_value"
                #param2="some_value"  # You can set param2 to a constant value
                
                # Run the function in the background
                #my_function "$param1" & #"$param2" &
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
fi



# # Iterate over the values and run the function concurrently
# for current_value in $(seq $start $step $end); do
#     param1="$current_value"
#     #param2="some_value"  # You can set param2 to a constant value
    
#     # Run the function in the background
#     #my_function "$param1" & #"$param2" &
#     compute_sound "$param1" "$gpu_index" &
#     gpu_index=$(((gpu_index+1)%total_number_of_available_gpus))
# done

# # Wait for all background processes to complete
# wait

# echo "All functions have completed"
