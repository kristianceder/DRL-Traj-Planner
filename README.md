# DDPG-Boosted MPC for Collision-Free Navigation of Multiple Mobile Robots
*Trajectory Planning of Multiple Robots using Vision-Based Continuous Deep Reinforcement Learning and Model Predictive Control*

![Example](doc/cover.jpg "Example")


**Publications:**
- [2023] [Collision-Free Trajectory Planning of Mobile Robots by Integrating Deep Reinforcement Learning and Model Predictive Control](https://ieeexplore.ieee.org/abstract/document/10260515), *CASE*
- [2024] [Trajectory Planning of Multiple Robots using Vision-Based Continuous Deep Reinforcement Learning and Model Predictive Control](https://github.com/kristianceder/DRL-Traj-Planner/blob/main/doc/iros24_official.pdf), *iROS*
- [2024] [Sample-Efficient Curriculum Reinforcement Learning for Complex Reward Functions](https://arxiv.org/abs/2410.16790), *Preprint*


The main branch of this repository is the single robot implementation. The multiple robot implementation is in a separate branch in the repository.

## Quick Start
### OpEn
The NMPC formulation is solved using open source implementation of PANOC, namely [OpEn](https://alphaville.github.io/optimization-engine/). Follow the [installation instructions](https://alphaville.github.io/optimization-engine/docs/installation) before proceeding. 

### Install dependencies (after installing OpEn)

Create a conda env with
```
conda env create -f environment.yaml
```
**NOTE** If you cannot create the virtual environment via conda, please create your own virtual environment as
```
conda create -n drl-traj-plan python=3.11
```
After that install the missing packages by
```
conda activate drl-traj-plan
pip install -r requirements.txt
```
Make sure your RUST is up-to-date and Pytorch is compatible with Cuda if you want to use a GPU. 

### Generate MPC solver
Go to "test_block_mpc.py", change **INIT_BUILD** to true and run
```
python test_block_mpc.py
```
After this, a new directory *mpc_build* will appear and contain the solver. Then, you are good to go :)

### Train RL models
To train a RL model, run the train.py script in the src folder as
```
python train.py
```
If packages and files are not found such as configs.py then run the following
```
export PYTHONPATH=/my/path/to/DRL-Traj-Planner/src
```
and replace the path with your actual path.

This should start a training and shows the success rate. If you want more detailed logging, activate use_wandb in [configs.py](./src/configs.py). You will find many more configs to change such as the map to train on and what algorithm to use.

To visualize your trained model run
```
python visualize.py
```
This will automatically pick the latest trained model and show you how it behaves. If you want another model, then you can specify it with the command line argument *-p* or *--path*.

**The following evaluation is work in progress and might not work as expected.**  
Finally, policies can be evaluated standalone (i.e. without MPC) or together with 
```
python evaluate_rl.py
python evaluate_hybrid.py
```

## Evaluation videos
Videos of the evaluations are available on [youtube](https://www.youtube.com/watch?v=A2TAuWXqH2k&list=PLPOBQOuy0QPRkYZ1olWVMvoB_gpyaQp5T)

