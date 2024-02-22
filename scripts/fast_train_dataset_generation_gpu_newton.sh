#!/bin/bash

NUM_NODES=1
NUM_CORES=100
NUM_GPUS=1
JOB_NAME="fast_gpu"
MAIL_USER="gabrieles@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/mambaforge
CONDA_ENV=backup

user=${1:-$USER}
echo "using user $user"

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
     --exclude=lambda5,bruno2,euler1,nlp-ada-1 \
	-o 'slurm-%N-%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env 
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV
echo "*** environment $CONDA_ENV activated! ***"

python EARS/forward_model/compute_sound.py \
    #  -rir_path "/home/$user/EARS/data/pressure_field_orientation_dataset/32_angles/default_5.0_5.0_order_1_0.5_d_0.05/rir/rir_indoor/" \
     -rir_path "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/mega_dataset/default_5.0_5.0_order_1_0.5_d_0.05/" \
	 -batch_size 300 -gpu 0

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF