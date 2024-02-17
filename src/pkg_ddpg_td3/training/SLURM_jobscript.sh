#!/usr/bin/env bash
#SBATCH -A C3SE2024-1-13                    # find your project with the "projinfo" command
#SBATCH -t 4-12:00:00                       # how long time it will take to run
#SBATCH -n 32                               # Number of cores
#SBATCH --gpus-per-node=T4:1                # choosing no. GPUs and their type
#SBATCH -J drl-var0-run9                    # the jobname

module purge

REPO_NAME=DRL-Traj-Planner
REPO_PATH=~/github/$REPO_NAME
CONTAINER=src/pkg_ddpg_td3/training/container.sif
TRAINING_SCRIPT=src/continous_training.py
MODEL=0
RUN=9


cp -r $REPO_PATH $TMPDIR
( cd $TMPDIR/$REPO_NAME ; apptainer exec ./$CONTAINER python $TRAINING_SCRIPT $MODEL $RUN )
mv $REPO_PATH/src/pkg_ddpg_td3/training/slurm-$SLURM_JOB_ID.out $REPO_PATH/Model/training/variant-$MODEL/run$RUN/slurm-run$RUN.out