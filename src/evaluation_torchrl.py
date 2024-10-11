### System import
import os
import pathlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

### DRL import
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from pkg_torchrl.sac import SAC
from pkg_torchrl.env import make_env
from torchrl.envs import step_mdp

from pkg_ddpg_td3.environment.environment import TrajectoryPlannerEnvironment

### MPC import
from interface_mpc import InterfaceMpc
from util.mpc_config import Configurator

### Helper
from main_pre_continous import generate_map, get_geometric_map, Metrics
from pkg_ddpg_td3.utils.map import test_scene_1_dict, test_scene_2_dict
#from pkg_dqn.utils.map import test_scene_1_dict, test_scene_2_dict

### Others
from timer import PieceTimer, LoopTimer
from typing import List, Tuple

from pkg_ddpg_td3.utils.map_eval import *
from configs import BaseConfig

MAX_RUN_STEP = 200
DYN_OBS_SIZE = 0.8 + 0.8

def ref_traj_filter(original: np.ndarray, new: np.ndarray, decay=1):
    filtered = original.copy()
    for i in range(filtered.shape[0]):
        filtered[i, :] = (1-decay) * filtered[i, :] + decay * new[i, :]
        decay *= decay
        if decay < 1e-2:
            decay = 0.0
    return filtered

def load_rl_model_env(generate_map, index: int) -> Tuple[SAC, SAC, TrajectoryPlannerEnvironment]:
    config = BaseConfig()
    models_folder = Path("../Model/testing")
    model_path_base = models_folder / "base_01" / "final_model.pth"
    model_path_cr = models_folder / "cr_1" / "final_model.pth"
    
    env_eval = make_env(config, generate_map=generate_map)
    base_model = SAC(config.sac, env_eval, env_eval)
    base_model.load(model_path_base)
    cr_model = SAC(config.sac, env_eval, env_eval)
    cr_model.load(model_path_cr)

    base_model, cr_model = base_model, cr_model

    return base_model, cr_model, env_eval

def load_mpc(config_path: str, verbose: bool = True):
    config = Configurator(config_path, verbose=verbose)
    traj_gen = InterfaceMpc(config, motion_model=None) # default motion model is used
    return traj_gen

def est_dyn_obs_positions(last_pos: list, current_pos: list, steps:int=20):
    """
    Estimate the dynamic obstacle positions in the future.
    """
    est_pos = []
    d_pos = [current_pos[0]-last_pos[0], current_pos[1]-last_pos[1]]
    for i in range(steps):
        est_pos.append([current_pos[0]+d_pos[0]*(i+1), current_pos[1]+d_pos[1]*(i+1), DYN_OBS_SIZE, DYN_OBS_SIZE, 0, 1])
    return est_pos

def circle_to_rect(pos: list, radius:float=DYN_OBS_SIZE):
    """
    Convert the circle to a rectangle.
    """
    return [[pos[0]-radius, pos[1]-radius], [pos[0]+radius, pos[1]-radius], [pos[0]+radius, pos[1]+radius], [pos[0]-radius, pos[1]+radius]]


def main_process(rl_index:int=1, decision_mode:int=1, to_plot=False, scene_option:Tuple[int, int, int]=(1, 1, 1), verbose:bool=False):
    """
    Args:
        rl_index: 0 for image, 1 for ray
        decision_mode: 0 for pure rl, 1 for pure mpc, 2 for hybrid
    """
    if verbose:
        prt_decision_mode = {0: 'pure_mpc', 1: 'base', 2: 'cr'}
        print(f"The decision mode is: {prt_decision_mode[decision_mode]}")

    time_list = []

    gen_map = eval(f"generate_eval_map{''.join(map(str, scene_option))}")
    base_model, cr_model, env_eval = load_rl_model_env(gen_map, rl_index)

    CONFIG_FN = 'mpc_longiter.yaml'
    cfg_fpath = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', CONFIG_FN)
    traj_gen = load_mpc(cfg_fpath, verbose=verbose)
    geo_map = get_geometric_map(env_eval.unwrapped.get_map_description(), inflate_margin=0.7)
    traj_gen.update_static_constraints(geo_map.processed_obstacle_list) # assuming static obstacles not changed

    done = False
    with torch.no_grad():
        while not done:
            state = env_eval.reset()

            init_state = np.array([*env_eval.unwrapped.agent.position, env_eval.unwrapped.agent.angle])
            goal_state = np.array([*env_eval.unwrapped.goal.position, 0])
            ref_path = list(env_eval.unwrapped.path.coords)
            traj_gen.initialization(init_state, goal_state, ref_path)

            last_mpc_time = 0.0
            last_rl_time = 0.0

            chosen_ref_traj = None
            rl_ref = None  
            last_dyn_obstacle_list = None            

            for i in range(0, MAX_RUN_STEP):

                print(f"\r{decision_mode}, {i+1}/{MAX_RUN_STEP}", end="  ")

                dyn_obstacle_list = [obs.keyframe.position.tolist() for obs in env_eval.unwrapped.obstacles if not obs.is_static]
                dyn_obstacle_tmp  = [obs+[DYN_OBS_SIZE, DYN_OBS_SIZE, 0, 1] for obs in dyn_obstacle_list]
                dyn_obstacle_list_poly = [circle_to_rect(obs) for obs in dyn_obstacle_list]
                dyn_obstacle_pred_list = []
                if last_dyn_obstacle_list is None:
                    last_dyn_obstacle_list = dyn_obstacle_list
                for j, dyn_obs in enumerate(dyn_obstacle_list):
                    dyn_obstacle_pred_list.append(est_dyn_obs_positions(last_dyn_obstacle_list[j], dyn_obs))
                last_dyn_obstacle_list = dyn_obstacle_list

                if decision_mode == 0:
                    env_eval.unwrapped.set_agent_state(traj_gen.state[:2], traj_gen.state[2], 
                                                traj_gen.last_action[0], traj_gen.last_action[1])
                    rand_action = env_eval.rand_action(state)
                    next_state = env_eval.step(rand_action) # just for plotting and updating status

                    if dyn_obstacle_list:
                        traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                    original_ref_traj, _ = traj_gen.get_local_ref_traj()
                    chosen_ref_traj = original_ref_traj
                    timer_mpc = PieceTimer()
                    try:
                        mpc_output = traj_gen.get_action(chosen_ref_traj)
                    except Exception as e:
                        done = True
                        print(f'MPC fails: {e}')
                        break
                    last_mpc_time = timer_mpc(4, ms=True)
                    if mpc_output is None:
                        break
                    action, pred_states, cost = mpc_output

                elif decision_mode == 1:
                    traj_gen.set_current_state(env_eval.unwrapped.agent.state)
                    original_ref_traj, *_ = traj_gen.get_local_ref_traj() # just for output

                    timer_rl = PieceTimer()
                    # td_action = base_model.predict(state, deterministic=True)
                    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                        td_action = base_model.model["policy"](state)
                    last_rl_time = timer_rl(4, ms=True)
                    next_state = env_eval.step(td_action)
                    done = next_state["next"]["done"].cpu().numpy()

                elif decision_mode == 2:
                    traj_gen.set_current_state(env_eval.unwrapped.agent.state)
                    original_ref_traj, *_ = traj_gen.get_local_ref_traj() # just for output

                    timer_rl = PieceTimer()
                    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                        td_action = cr_model.model["policy"](state)
                    last_rl_time = timer_rl(4, ms=True)
                    next_state = env_eval.step(td_action)
                    done = next_state["next"]["done"].cpu().numpy()

                else:
                    raise ValueError("Invalid decision mode")
                
                if decision_mode == 0:
                    time_list.append(last_mpc_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (MPC): {last_mpc_time}ms")
                elif decision_mode == 1:
                    time_list.append(last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (Baseline): {last_rl_time}ms")
                elif decision_mode == 2:
                    time_list.append(last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (Curriculum): {last_rl_time}ms")     


                if to_plot & (i%1==0): # render every third frame
                    env_eval.unwrapped.render(dqn_ref=rl_ref, actual_ref=chosen_ref_traj)

                if i == MAX_RUN_STEP - 1:
                    done = True
                    if verbose:
                        print('Time out!')
                if done:
                    if to_plot:
                        input('Collision or finish! Press enter to continue...')
                    break
                
                state = step_mdp(next_state)

    action_list = [(v, w) for (v, w) in zip(env_eval.unwrapped.speeds, env_eval.unwrapped.angular_velocities)]

    success = next_state["next"]["success"].cpu().numpy()

    if verbose:
        print(f"Average time ({prt_decision_mode[decision_mode]}): {np.mean(time_list)}ms\n")
    else:
        print()
    return time_list, success, action_list, traj_gen.ref_traj, env_eval.unwrapped.traversed_positions, geo_map.obstacle_list

def main_evaluate(rl_index: int, decision_mode, metrics: Metrics, scene_option:Tuple[int, int, int]) -> Metrics:
    to_plot = True
    time_list, success, actions, ref_traj, actual_traj, obstacle_list = main_process(rl_index=rl_index,
                                                                                     decision_mode=decision_mode,
                                                                                     to_plot=to_plot,
                                                                                     scene_option=scene_option,
                                                                                     verbose=True)
    metrics.add_trial_result(computation_time_list=time_list, succeed=success, action_list=actions, 
                             ref_trajectory=ref_traj, actual_trajectory=actual_traj, obstacle_list=obstacle_list)
    return metrics


if __name__ == '__main__':
    """
    rl_index: 0: image, 1: ray
    decision_mode: 0: mpc, 1: ddpg, 2: td3, 3: hybrid-ddpg

    Map:
    SCENE 1:
    - 1: Single rectangular static obstacle 
        - (1-small, 2-medium, 3-large)
    - 2: Two rectangular static obstacles 
        - (1-small stagger, 2-large stagger, 3-close aligned, 4-far aligned)
    - 3: Single non-convex static obstacle
        - (1-big u-shape, 2-small u-shape, 3-big v-shape, 4-small v-shape)
    - 4: Single dynamic obstacle
        - (1-crash, 2-cross)

    SCENE 2:
    - 1: Single rectangular obstacle
        - (1-right, 2-sharp, 3-u-shape)
    - 2: Single dynamic obstacle
        - (1-right, 2-sharp, 3-u-shape)

    rl_index: 0 = image, 1 = ray
    decision_mode: 0 = MPC, 1 = Baseline, 2 = Curriculum
    """
    num_trials = 1 # 50
    print_latex = True
    scene_option_list = [
                        #  (1, 1, 1), # a-small
                        #  (1, 1, 2), # a-medium
                        #  (1, 1, 3), # b-large
                        #  (1, 2, 1), # c-small
                        #  (1, 2, 2), # d-large
                        #  (1, 2, 3), # d-large
                        #  (1, 2, 4), # d-large
                        #  (1, 3, 1), # e-small
                        #  (1, 3, 2), # f-large
                        #  (1, 3, 3), # ?
                        #  (1, 3, 4), # ?
                        #  (1, 4, 1), # face-to-face
                        #  (1, 4, 2), # ?
                        #  (1, 4, 3), # ?
                        # (1, 5, 1) # eval map long
                        (1, 5, 2) # eval map
                         ]
                        #  (2, 1, 1), # right turn with an obstacle
                        #  (2, 1, 2), # sharp turn with an obstacle
                        #  (2, 1, 3), # u-turn with an obstacle

    for scene_option in scene_option_list:

        print(f"=== Scene {scene_option[0]}-{scene_option[1]}-{scene_option[2]} ===")

        mpc_metrics = Metrics(mode='MPC')
        baseline_metrics = Metrics(mode='Baseline')
        cr_metrics = Metrics(mode='Curriculum')

        for i in range(num_trials):
            print(f"Trial {i+1}/{num_trials}")
            # mpc_metrics = main_evaluate(rl_index=1, decision_mode=0, metrics=mpc_metrics, scene_option=scene_option)
            baseline_metrics = main_evaluate(rl_index=0, decision_mode=1, metrics=baseline_metrics, scene_option=scene_option)
            cr_metrics = main_evaluate(rl_index=0, decision_mode=2, metrics=cr_metrics, scene_option=scene_option)

        round_digits = 2
        print(f"=== Scene {scene_option[0]}-{scene_option[1]}-{scene_option[2]} ===")
        print('Baseline')
        print(baseline_metrics.get_average(round_digits))
        print()
        print('Curriculum')
        print(cr_metrics.get_average(round_digits))
        print()


        ## Write to latex
        if print_latex:
            print(f"=== Scene {scene_option[0]}-{scene_option[1]}-{scene_option[2]} ===")
            print(baseline_metrics.write_latex(round_digits))
            print(cr_metrics.write_latex(round_digits))