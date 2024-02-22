#!/bin/bash

process_data() {
    which_gpu="$1"
    chunk_index="$2"
    total_number_of_available_gpus="$3"
    n_angles="$4"

    echo "working on $which_gpu with chunk index $chunk_index (total available $total_number_of_available_gpus)" 
    
    python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
      -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
      -second_rad 0.51 -opt_harmonies 0.5 1 2 3 \
      -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 \
      -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 4 \
      -num_phases_mics 8 -saved_data_dir_path default \
         -phase_shift 0 0 0 0 -e_absorption 0.5 -delta 0.05 \
         -num_points_per_side 63 -room_x 5.0 -room_y 5.0 \
         -flip_rotation_direction 1 0 0 1 \
         -gpu "$which_gpu" -chunk "$chunk_index" -total_gpus_available "$total_number_of_available_gpus"\
         -compute_sound_only -number_of_angles "$n_angles" -save_in_mega

   echo "FINISHED working on $which_gpu with chunk index $chunk_index (total available $total_number_of_available_gpus)" 
}

# Check if at least one parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <number of angles to generate per position>"
    exit 1
fi

number_of_angles=$1

while true; do
   ## GET GPU STATUS INFORMATION

   # Your GPU status information
   gpu_status=$(gpustat)

   # Extract GPU information
   gpu_info=$(echo "$gpu_status" | grep -oP '\[\K[0-9]+.*?%')

   # Initialize an array to store maps
   declare -A gpu_array

   # Initialize counter
   available_gpu_count=0
   # Initialize an array to store available GPU indices
   available_gpus=()

   # Iterate over GPU information and populate the array
   while read -r gpu; do
      index=$(echo "$gpu" | cut -d' ' -f1 | tr -d '[]')
      percentage=$(echo "$gpu" | grep -oP '[0-9]+(?=\s%)')
      gpu_array["$index"]=$percentage

      # Check percentage and add index to the available_gpus array
      if [ "$percentage" -lt 20 ]; then
         available_gpus+=("$index")
         ((available_gpu_count++))
      fi
   done <<< "$gpu_info"

   # Sort the available_gpus array
   IFS=$'\n' sorted_available_gpus=($(sort <<<"${available_gpus[*]}"))

   chunk_counter=0
   # Loop over the sorted available GPU indices
   for index in "${sorted_available_gpus[@]}"; do
      process_data "$index" "$chunk_counter" "$available_gpu_count" "$number_of_angles" &
      ((chunk_counter++))
   done
   
   wait

   sleep 120 # wait for 2 minutes before starting over again 

done