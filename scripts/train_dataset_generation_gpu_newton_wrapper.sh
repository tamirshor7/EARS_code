#!/bin/bash

# Check if at least one parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <total number of gpus> <[OPTIONAL] user>"
    exit 1
fi

total_number_of_available_gpus=$1
user=${2:-$USER}

# Display the parameter
echo "The total number of available gpus is: $total_number_of_available_gpus"

for chunk in $(seq 0 1 $total_number_of_available_gpus); do
      ./EARS/scripts/train_dataset_generation_gpu_newton.sh $chunk $total_number_of_available_gpus $user &
done

wait

echo "The computation of the sound has been launched on all of the asked GPUs"