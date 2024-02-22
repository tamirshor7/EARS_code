# #!/bin/bash

# # Your GPU status information
# gpu_status=$(gpustat)

# # Extract GPU information
# gpu_info=$(echo "$gpu_status" | grep -oP '\[\K[0-9]+.*?%')

# # Initialize counter
# available_gpu_count=0

# # Initialize an array to store maps
# declare -A gpu_array

# # Initialize an array to store available GPU indices
# available_gpus=()

# # Iterate over GPU information and populate the array
# while read -r gpu; do
#     index=$(echo "$gpu" | cut -d' ' -f1 | tr -d '[]')
#     percentage=$(echo "$gpu" | grep -oP '[0-9]+(?=\s%)')
#     gpu_array["$index"]=$percentage

#     # Check percentage and add index to the available_gpus array
#     if [ "$percentage" -lt 20 ]; then
#         available_gpus+=("$index")
#         ((available_gpu_count++))
#     fi
# done <<< "$gpu_info"

# # Sort the available_gpus array
# IFS=$'\n' sorted_available_gpus=($(sort <<<"${available_gpus[*]}"))

# # Loop over the sorted available GPU indices
# for index in "${sorted_available_gpus[@]}"; do
#     percentage=${gpu_array[$index]}
#     echo "GPU $index is available with $percentage% memory"
# done

# echo "Total available GPUs: $available_gpu_count"


#!/bin/bash

# GPU information
gpu_info="juliet Sun Dec 10 16:18:40 2023 525.147.05 [0] NVIDIA GeForce RTX 3090 | 21'C, 0 % | 319 / 24576 MB | [1] NVIDIA GeForce RTX 3090 | 22'C, 0 % | 319 / 24576 MB | [2] NVIDIA GeForce RTX 3090 | 21'C, 0 % | 319 / 24576 MB | [3] NVIDIA GeForce RTX 3090 | 21'C, 0 % | 319 / 24576 MB | [4] NVIDIA GeForce RTX 3090 | 22'C, 0 % | 319 / 24576 MB | [5] NVIDIA GeForce RTX 3090 | 24'C, 0 % | 319 / 24576 MB | [6] NVIDIA GeForce RTX 3090 | 23'C, 0 % | 319 / 24576 MB | [7] NVIDIA GeForce RTX 3090 | 23'C, 0 % | 319 / 24576 MB |"

# Initialize array
declare -A gpu_array

# Parse GPU information
while IFS='|' read -r gpu_info_line; do
    gpu_index=$(echo "$gpu_info_line" | grep -oP '\[\K[0-9]+')
    total_memory=$(echo "$gpu_info_line" | grep -oP '\|\s[0-9]+\s/\s[0-9]+\sMB\s\|' | awk -F'/' '{print $2}' | tr -d '[:space:]')
    used_memory=$(echo "$gpu_info_line" | grep -oP '\|\s[0-9]+\s/\s[0-9]+\sMB\s\|' | awk -F'/' '{print $1}' | tr -d '[:space:]')

    # Check if total_memory and used_memory are numeric
    if [[ "$total_memory" =~ ^[0-9]+$ && "$used_memory" =~ ^[0-9]+$ && "$total_memory" -ne 0 ]]; then
        percentage=$(echo "scale=2; ($used_memory / $total_memory) * 100" | bc)
    else
        percentage=0
    fi

    
    # Add to the array
    gpu_array["$gpu_index"]=$percentage
done <<< "$gpu_info"

# Print the array
for index in "${!gpu_array[@]}"; do
    echo "GPU $index: ${gpu_array[$index]}%"
done
echo "${gpu_array[@]}"
