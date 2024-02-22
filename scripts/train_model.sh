#!/bin/bash

NUM_NODES=1
NUM_CORES=4
NUM_GPUS=1
JOB_NAME="train"
MAIL_USER="gabrieles@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/mambaforge
CONDA_ENV=backup

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

python EARS/localization/train_pipeline.py --use-newton-cluster --batch-size 4 --use-multi-position --aggregator transformer

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF