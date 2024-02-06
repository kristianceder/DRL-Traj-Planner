#!/usr/bin/env bash
#SBATCH -A C3SE2024-1-13                    # find your project with the "projinfo" command
#SBATCH -t 3-00:00:00                       # how long time it will take to run
#SBATCH --array 0-7                         # Slurm array numbers. Enables several scripts to run in the same job.
#SBATCH -n 32                               # Number of cores
#SBATCH --gpus-per-node=T4:1                # choosing no. GPUs and their type
#SBATCH -J drl_obstacle_avoidance-test_job  # the jobname

module purge

REPO_NAME=TrajTrack_MPCnDQN_RLBoost_Design_Project_Group_1
REPO_PATH=~/$REPO_NAME
CONTAINER=src/pkg_ddpg_td3/training/container.sif
TRAINING_SCRIPT=src/continous_training.py


cp -r $REPO_PATH $TMPDIR
( cd $TMPDIR/$REPO_NAME ; apptainer exec ./$CONTAINER python $TRAINING_SCRIPT $SLURM_ARRAY_TASK_ID )

