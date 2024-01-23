# Training

Guide to start training a DDPG or TD3 model.

## Container

To successfully run training of either a DDPG or TD3 model on a cluster, a container created with the tools of "apptainer" with the correct dependencies is needed.
Since git does not allow such large files, a usable container can be aquired via this link: <br>
https://chalmers-my.sharepoint.com/:u:/g/personal/nymanga_chalmers_se/EQAjWrllmpxCimSCgBliZ7ABImpQlsYZXW9AHvzzDfYaoQ?e=3yV4ue <br>
You need to be logged in to a chalmers account to gain access. 
Move the container to this current folder (src/pkg_ddpg_t3/training) using for example scp. If the container is placed elsewhere, the variable CONTAINER in SLURM_jobcript.sh must be changed.

## Training script

The training is started by running the file src/continous_training.py. Changing the parameters used in the training is done in this file.

 Running the script takes one positional argument which corresponds to the "index" that defines the parameters used, as seen in the file. Having the index as a positional argument allows SLURM to run several trainings with different parameters within the same job as seen in the next section. The scrips can of course also be excecuted directly from the command line with for example:<br>
 *./src/continous_training.py 0*<br>
 To run training with index 0.

**IMPORTANT**: Change the variable "path" in the "run()" function to the directory where you would like to store the created models.

## Jobscript

The file src/pkg_ddpg_t3/training/SLURM_jobscript.sh is a bash template for how to submit a training job in SLURM. The job is submitted by writing:<br>
*sbatch src/pkg_ddpg_td3/training/SLURM_jobscript.sh*<br>
in the terminal.

### Variables

The file contains definitions that needs to be changed if the structure is changed.
- REPO_NAME - The chosen name of the main folder
- REPO_PATH - The path to the location of the folder
- CONTAINER - Path to the apptainer container
- TRAINING_SCRIPT - Path to the script that executes the training 


### SLURM flags

The lines starting with "#SBATCH ..." in the beginning of the file defines the arguments that SLURM takes.

- -A - The name of the project the job uses for billing. Your available projects can be found with the "projinfo" command.
- -t - The maximum time the job will take to execute. If the max time is reached before the job is finished, it will be terminated. Used to determine priority. Format: "days-hours:minutes:seconds".
- --array - Allows several trainings to be performed within the same job. Can be defined as a span (0-7) or discrete numbers (1,4,5). Generates a token called $SLURM_ARRAY_TASK_ID which corresponds to the array/subtask number. The number is then fed to the array script as the "index" and uses the corresponding parameters.
- -n - The number of CPU cores to be allocated
- --gpus-per-node - The type of GPU and number of GPUs to be used respectively. Example: T4:1 or A40:4.
- -J - Name of the project to be shown in for example the "projinfo" list.

### Excecution

The script simply initializes the variables mentioned above, copies the repository to the target node by using the built in variable for the nodes internal storage $TMPDIR, then executes the training script. 

- cd $TMPDIR/$REPO_NAME ; - Used to ensure that the script is executed from the node.
- apptainer exec ./$CONTAINER - Executes command using the container as environment.
- python $TRAINING_SCRIPT $SLURM_ARRAY_TASK_ID - Runs the script with the array number as index.



