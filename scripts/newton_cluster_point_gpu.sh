#!/bin/bash

NUM_NODES=1
NUM_CORES=4
NUM_GPUS=1
JOB_NAME="gpu_dataset_computation"
MAIL_USER="gabrieles@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/mambaforge
CONDA_ENV=backup

cluster_center_x=$1
cluster_center_y=$2
echo "using cluster center $cluster_center_x $cluster_center_y"

user=${2:-$USER}
echo "using user $user"

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-%N-%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env 
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV
echo "*** environment $CONDA_ENV activated! ***"


python EARS/forward_model/forward_indoor_wrapper.py -max_order 1 \
 -exp_name Dec20_13-33-20_plato1_final_rads_80mics_128sources_051 \
 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 \
 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 \
 -num_sources_power 7 -duration 0.5 -channel 0 -num_rotors 4 \
 -num_phases_mics 8 -saved_data_dir_path default \
    -phase_shift 0 0 0 0 -e_absorption 0.5 -delta 0.05 \
    -num_points_per_side 63 -room_x 5.0 -room_y 5.0 \
    -flip_rotation_direction 1 0 0 1 -gpu -1 \
    -number_of_angles 64 -compute_sound_only -use_newton_cluster \
	-compress_data -user $user \
	-chunk 0 -total_gpus_available 1 \
    -cluster_around_point --cluster_center $cluster_center_x $cluster_center_y

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF