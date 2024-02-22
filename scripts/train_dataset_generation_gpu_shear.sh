#!/bin/bash

process_data() {
    which_gpu="$1"
    chunk_index="$2"
    total_number_of_available_gpus="$3"

    echo "working on $which_gpu with chunk index $chunk_index (total available $total_number_of_available_gpus)"

    # Define the range of values
      start=45
      end=90
      step=5

      # Iterate over the values and run the function concurrently
      # for current_value in $(seq $start $step $end); do
      #    param1="$current_value"
         
      #       python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
      #       -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
      #       -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
      #       -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
      #       -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "rir_shear_deformation_$param1" \
      #       -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 \
      #       -room_x 5.0 -room_y 5.0 -margins 0.02 -rotating_angle "$param1" \
      #       -flip_rotation_direction 1 0 0 1  -number_of_angles 2 \
      #       -gpu "$which_gpu" -chunk "$chunk_index" -total_gpus_available "$total_number_of_available_gpus"\
      #       -compute_sound_only &
      # done


      python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
            -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
            -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
            -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
            -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "rir_shear_deformation_45" \
            -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 \
            -room_x 5.0 -room_y 5.0 -margins 0.02 -rotating_angle "45" \
            -flip_rotation_direction 1 0 0 1  -number_of_angles 2 \
            -gpu "$which_gpu" -chunk "$chunk_index" -total_gpus_available "$total_number_of_available_gpus"\
            -compute_sound_only >> shear_progress_45.out &
      pid1=$!
      
      python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
            -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
            -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
            -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
            -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "rir_shear_deformation_50" \
            -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 \
            -room_x 5.0 -room_y 5.0 -margins 0.02 -rotating_angle "50" \
            -flip_rotation_direction 1 0 0 1  -number_of_angles 2 \
            -gpu "$which_gpu" -chunk "$chunk_index" -total_gpus_available "$total_number_of_available_gpus"\
            -compute_sound_only >> shear_progress_50.out &
      pid2=$!
      
      python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
            -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
            -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
            -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
            -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "rir_shear_deformation_55" \
            -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 \
            -room_x 5.0 -room_y 5.0 -margins 0.02 -rotating_angle "55" \
            -flip_rotation_direction 1 0 0 1  -number_of_angles 2 \
            -gpu "$which_gpu" -chunk "$chunk_index" -total_gpus_available "$total_number_of_available_gpus"\
            -compute_sound_only >> shear_progress_55.out  &
      pid3=$!
      
      python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
            -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
            -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
            -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
            -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "rir_shear_deformation_60" \
            -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 \
            -room_x 5.0 -room_y 5.0 -margins 0.02 -rotating_angle "60" \
            -flip_rotation_direction 1 0 0 1  -number_of_angles 2 \
            -gpu "$which_gpu" -chunk "$chunk_index" -total_gpus_available "$total_number_of_available_gpus"\
            -compute_sound_only >> shear_progress_60.out &
      pid4=$!
      
      python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
            -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
            -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
            -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
            -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "rir_shear_deformation_65" \
            -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 \
            -room_x 5.0 -room_y 5.0 -margins 0.02 -rotating_angle "65" \
            -flip_rotation_direction 1 0 0 1  -number_of_angles 2 \
            -gpu "$which_gpu" -chunk "$chunk_index" -total_gpus_available "$total_number_of_available_gpus"\
            -compute_sound_only >> shear_progress_65.out  &
      pid5=$!

      wait $pid1
      wait $pid2
      wait $pid3
      wait $pid4
      wait $pid5

      python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
            -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
            -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
            -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
            -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "rir_shear_deformation_70" \
            -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 \
            -room_x 5.0 -room_y 5.0 -margins 0.02 -rotating_angle "70" \
            -flip_rotation_direction 1 0 0 1  -number_of_angles 2 \
            -gpu "$which_gpu" -chunk "$chunk_index" -total_gpus_available "$total_number_of_available_gpus"\
            -compute_sound_only >> shear_progress_70.out &
      pid1=$!
      
      python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
            -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
            -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
            -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
            -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "rir_shear_deformation_75" \
            -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 \
            -room_x 5.0 -room_y 5.0 -margins 0.02 -rotating_angle "75" \
            -flip_rotation_direction 1 0 0 1  -number_of_angles 2 \
            -gpu "$which_gpu" -chunk "$chunk_index" -total_gpus_available "$total_number_of_available_gpus"\
            -compute_sound_only >> shear_progress_75.out &
      pid2=$!
      
      python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
            -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
            -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
            -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
            -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "rir_shear_deformation_80" \
            -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 \
            -room_x 5.0 -room_y 5.0 -margins 0.02 -rotating_angle "80" \
            -flip_rotation_direction 1 0 0 1  -number_of_angles 2 \
            -gpu "$which_gpu" -chunk "$chunk_index" -total_gpus_available "$total_number_of_available_gpus"\
            -compute_sound_only >> shear_progress_80.out &
      pid3=$!
      
      python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
            -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
            -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
            -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
            -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "rir_shear_deformation_85" \
            -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 \
            -room_x 5.0 -room_y 5.0 -margins 0.02 -rotating_angle "85" \
            -flip_rotation_direction 1 0 0 1  -number_of_angles 2 \
            -gpu "$which_gpu" -chunk "$chunk_index" -total_gpus_available "$total_number_of_available_gpus"\
            -compute_sound_only >> shear_progress_85.out &
      pid4=$!
      
      python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
            -exp_name Sep19_10-23-51_aida_new_rads_80mics_128sources_051 \
            -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 \
            -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 \
            -num_rotors 4 -num_phases_mics 8 -saved_data_dir_path "rir_shear_deformation_90" \
            -phase_shift 0 0 0 0 -e_absorption 0.5 -num_points_per_side 25 \
            -room_x 5.0 -room_y 5.0 -margins 0.02 -rotating_angle "90" \
            -flip_rotation_direction 1 0 0 1  -number_of_angles 2 \
            -gpu "$which_gpu" -chunk "$chunk_index" -total_gpus_available "$total_number_of_available_gpus"\
            -compute_sound_only >> shear_progress_90.out &
      pid5=$!

      wait $pid1
      wait $pid2
      wait $pid3
      wait $pid4
      wait $pid5
    

   echo "FINISHED working on $which_gpu with chunk index $chunk_index (total available $total_number_of_available_gpus)" 
}

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
      process_data "$index" "$chunk_counter" "$available_gpu_count" &
      ((chunk_counter++))
   done
   
   wait

   sleep 120 # wait for 2 minutes before starting over again 

done