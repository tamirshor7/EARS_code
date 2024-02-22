#!/bin/bash

# Define your function with parameters
my_function() {
    param1="$1"
    #param2="$2"
    
    room_x=$(echo "5 * $param1" | bc)
    #room_y=$(echo "5 * $param1" | bc)
    delta=$(echo "0.25 * $param1" | bc)

    # Your function logic here
    echo "Running function with param1=$param1"
    
    nohup python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
     -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 \
    -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
     -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
     -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path non_uniform \
     -phase_shift 0 0 0 0 -e_absorption 0.5 -delta "$delta" \
     -room_x "$room_x" -room_y 5.0 > $param1_non_uniform.out 
    
    echo "Function with param1=$param1"
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

# Define the range of values
start=0.5
end=2.0
step=0.1

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
