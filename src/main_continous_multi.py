### System import
import os
import copy
import pathlib
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
from matplotlib.patches import Patch # type: ignore
from matplotlib.lines import Line2D # type: ignore

### DRL import
import gym # type: ignore
from torch import no_grad
from stable_baselines3 import DDPG, TD3
from pkg_ddpg_td3.utils.per_ddpg import PerDDPG
from stable_baselines3.common import env_checker

from pkg_ddpg_td3.environment import MobileRobot, Obstacle
from pkg_ddpg_td3.environment.environment import TrajectoryPlannerEnvironment

### MPC import
from interface_mpc import InterfaceMpc
from util.mpc_config import Configurator

### Helper
from main_pre_continous_multi import generate_map, get_geometric_map, HintSwitcher, Metrics
from pkg_ddpg_td3.utils.map import test_scene_1_dict, test_scene_2_dict
#from pkg_dqn.utils.map import test_scene_1_dict, test_scene_2_dict

### Others
from robot_manager import RobotManager
from timer import PieceTimer, LoopTimer


MAX_RUN_STEP = 200
DYN_OBS_SIZE = 0.8 + 0.8

def ref_traj_filter(original: np.ndarray, new: np.ndarray, decay=1.0):
    filtered = original.copy()
    for i in range(filtered.shape[0]):
        filtered[i, :] = (1-decay) * filtered[i, :] + decay * new[i, :]
        decay *= decay
        if decay < 1e-2:
            decay = 0.0
    return filtered

def load_rl_model_env(generate_map_list: list, index: int) -> Tuple[PerDDPG, TD3, List[TrajectoryPlannerEnvironment]]:
    variant = [
        {
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward1-v0',
            'net_arch': [64, 64],
            'per': False,
            'device': 'auto',
        },
        {
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [16, 16],
            'per': False,
            'device': 'cpu',
        },
    ][index] 

    if index == 0:
        model_folder_name = 'image'
    elif index == 1:
        model_folder_name = 'ray'
    else:
        raise ValueError('Invalid index')
    model_path_ddpg = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'Model/ddpg', model_folder_name, 'best_model')
    model_path_td3 = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'Model/td3', model_folder_name, 'best_model')
    
    env_list = []
    for gen_map_agent in generate_map_list:
        env_eval:TrajectoryPlannerEnvironment = gym.make(variant['env_name'], generate_map=gen_map_agent)
        env_checker.check_env(env_eval)
        env_list.append(env_eval)
    td3_model = TD3.load(model_path_td3)
    ddpg_model = PerDDPG.load(model_path_ddpg)
    return ddpg_model, td3_model, env_list

def load_mpc(config: Configurator):
    traj_gen = InterfaceMpc(config, motion_model=None) # default motion model is used
    return traj_gen

def est_dyn_obs_positions(last_pos: list, current_pos: list, steps=20):
    """Estimate the dynamic obstacle positions in the future.
    """
    est_pos = []
    d_pos = [current_pos[0]-last_pos[0], current_pos[1]-last_pos[1]]
    for i in range(steps):
        est_pos.append([current_pos[0]+d_pos[0]*(i+1), current_pos[1]+d_pos[1]*(i+1), DYN_OBS_SIZE, DYN_OBS_SIZE, 0, 1])
    return est_pos

def circle_to_rect(pos: list, radius:float=DYN_OBS_SIZE):
    """Convert the circle to a rectangle.
    """
    return [[pos[0]-radius, pos[1]-radius], [pos[0]+radius, pos[1]-radius], [pos[0]+radius, pos[1]+radius], [pos[0]-radius, pos[1]+radius]]


def main(rl_index:int=1, decision_mode:int=1, to_plot=False, map_only=False, scene_option:int=1, save=False):
    """
    Args:
        rl_index: 0 for image, 1 for ray
        decision_mode: 0 for pure MPC, 1 for pure DDPG, 2 for pure TD3, 3 for hybrid DDPG, 4 for hybrid TD3
    """
    prt_decision_mode = {0: 'pure_mpc', 1: 'pure_ddpg', 2: 'pure_td3', 3: 'hybrid_ddpg', 4: 'hybrid_td3'}
    print(f"The decision mode is: {prt_decision_mode[decision_mode]}")

    time_list = []

    ddpg_model, td3_model, env_eval_list = load_rl_model_env(generate_map(scene_option), rl_index)
    num_robots = len(env_eval_list)

    CONFIG_FN = 'mpc_longiter.yaml'
    # CONFIG_FN = 'mpc_default.yaml'
    cfg_fpath = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', CONFIG_FN)
    config_mpc = Configurator(cfg_fpath)
    traj_gen_list = [load_mpc(config_mpc) for _ in range(num_robots)]
    geo_map = get_geometric_map(env_eval_list[0].get_map_description(), inflate_margin=0.8)
    for traj_gen in traj_gen_list:
        traj_gen.update_static_constraints(geo_map.processed_obstacle_list) # if assuming static obstacles not changed

    robot_manager = RobotManager(num_robots=num_robots, config_mpc=config_mpc)

    # 
    color_list = ['#2878b5', '#9ac9db', '#f8ac8c', '#c82423', '#bb9727', 
                  '#54b345', '#32b897', '#05b9e2', '#8983bf', '#c76da2',
                  '#f8ac8c', '#c82423', '#bb9727', '#54b345', '#32b897',]
    done_list = [False for _ in range(num_robots)]
    with no_grad():
        while not all(done_list):
            obsv_list = [env.reset() for env in env_eval_list]

            init_state_list = [np.array([*env.agent.position, env.agent.angle]) for env in env_eval_list]
            goal_state_list = [np.array([*env.goal.position, 0]) for env in env_eval_list]
            ref_path_list = [list(env.path.coords) for env in env_eval_list]

            for robot_idx in range(num_robots):
                robot_manager.add_robot(robot_idx)
                robot_manager.set_robot_state(robot_idx, init_state_list[robot_idx])
                robot_manager.set_pred_states(robot_idx, None)
                traj_gen_list[robot_idx].initialization(init_state_list[robot_idx], 
                                                        goal_state_list[robot_idx], 
                                                        ref_path_list[robot_idx])

            last_mpc_time = 0.0
            last_rl_time = 0.0

            chosen_ref_traj = None
            rl_ref = None
            last_dyn_obstacle_list = None      

            switch = HintSwitcher(20, 2, 10, always_on=False)

            if to_plot:
                
                if map_only:
                    fig, ax_main = plt.subplots(figsize=(10, 10))
                    fig.tight_layout()
                    ax_profile = None
                    ax_obsv = None
                    axes = [ax_main]
                else:
                    if rl_index == 0: # image
                        fig = plt.figure(figsize=(22, 6))
                        gs = gridspec.GridSpec(1, 3, width_ratios=[3, 2, 2])
                        axes = [fig.add_subplot(gs_) for gs_ in gs]
                        ax_main, ax_profile, ax_obsv = axes
                    else: # ray
                        fig = plt.figure(figsize=(16, 6))
                        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2])
                        axes = [fig.add_subplot(gs_) for gs_ in gs]
                        ax_main, ax_profile = axes
                        ax_obsv = None
                    fig.tight_layout()

                ax_main.set_aspect('equal')

            for i in range(0, MAX_RUN_STEP):
                
                ### Get dynamic obstacles
                dyn_obstacle_list = [obs.keyframe.position.tolist() for obs in env_eval_list[0].obstacles if not obs.is_static]
                dyn_obstacle_tmp  = [obs+[DYN_OBS_SIZE, DYN_OBS_SIZE, 0, 1] for obs in dyn_obstacle_list] # TODO check this
                dyn_obstacle_list_poly = [circle_to_rect(obs) for obs in dyn_obstacle_list]
                dyn_obstacle_pred_list = []
                if last_dyn_obstacle_list is None:
                    last_dyn_obstacle_list = dyn_obstacle_list
                for j, dyn_obs in enumerate(dyn_obstacle_list):
                    dyn_obstacle_pred_list.append(est_dyn_obs_positions(last_dyn_obstacle_list[j], dyn_obs))
                last_dyn_obstacle_list = dyn_obstacle_list

                ### Get all robot states
                robot_states = [robot_manager.get_robot_state(i) for i in range(num_robots)]
                robot_states_poly = [circle_to_rect(pos.tolist(), radius=config_mpc.vehicle_width) for pos in robot_states]
                robot_states_poly_obs = [Obstacle.create_mpc_static(obstacle) for obstacle in robot_states_poly]

                plot_map = True if scene_option != 0 else False
                plot_obsv = True
                if to_plot:
                    [ax.cla() for ax in axes]

                for robot_i in range(len(env_eval_list)):

                    obsv = obsv_list[robot_i]
                    traj_gen, env_eval = traj_gen_list[robot_i], env_eval_list[robot_i]
                    theme_color = color_list[robot_i]

                    if decision_mode == 0:
                        env_eval.set_agent_state(traj_gen.state[:2], traj_gen.state[2], 
                                                traj_gen.last_action[0], traj_gen.last_action[1])
                        obsv, reward, done, info = env_eval.step([0,0]) # just for plotting and updating status
                        obsv_list[robot_i] = obsv

                        if dyn_obstacle_list:
                            traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                        other_robot_states = robot_manager.get_other_robot_states(robot_i)
                        traj_gen.update_other_robot_states(other_robot_states)

                        original_ref_traj, *_ = traj_gen.get_local_ref_traj()
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
                            done = True
                        else:
                            action, pred_states, cost = mpc_output
                            robot_manager.set_pred_states(robot_i, pred_states)

                        robot_manager.set_robot_state(robot_i, traj_gen.state)

                    elif decision_mode == 1:
                        traj_gen.set_current_state(env_eval.agent.state)
                        original_ref_traj, *_ = traj_gen.get_local_ref_traj() # just for output

                        timer_rl = PieceTimer()
                        action_index, _states = ddpg_model.predict(obsv, deterministic=True)
                        last_rl_time = timer_rl(4, ms=True)
                        env_eval.set_temp_obstacles([x for i, x in enumerate(robot_states_poly_obs) if i != robot_i])
                        obsv, reward, done, info = env_eval.step(action_index)
                        
                        robot_manager.set_robot_state(robot_i, traj_gen.state)
                        robot_manager.set_pred_states(robot_i, None)

                    elif decision_mode == 2:
                        traj_gen.set_current_state(env_eval.agent.state)
                        original_ref_traj, *_ = traj_gen.get_local_ref_traj() # just for output

                        timer_rl = PieceTimer()
                        action_index, _states = td3_model.predict(obsv, deterministic=True)
                        last_rl_time = timer_rl(4, ms=True)
                        env_eval.set_temp_obstacles([x for i, x in enumerate(robot_states_poly_obs) if i != robot_i])
                        obsv, reward, done, info = env_eval.step(action_index)

                        robot_manager.set_robot_state(robot_i, traj_gen.state)
                        robot_manager.set_pred_states(robot_i, None)

                    elif decision_mode == 3:
                        env_eval.set_agent_state(traj_gen.state[:2], traj_gen.state[2], 
                                                traj_gen.last_action[0], traj_gen.last_action[1])
                        timer_rl = PieceTimer()
                        action_index, _states = ddpg_model.predict(obsv, deterministic=True)
                        env_eval.set_temp_obstacles([x for i, x in enumerate(robot_states_poly_obs) if i != robot_i])
                        # obsv, reward, done, info = env_eval.step(action_index)
                        ### Manual step
                        env_eval.step_obstacles()
                        env_eval.update_status(reset=False)
                        obsv = env_eval.get_observation()
                        done = env_eval.update_termination()
                        info = env_eval.get_info()

                        rl_ref = []
                        robot_sim:MobileRobot = copy.deepcopy(env_eval.agent)
                        for j in range(20):
                            if j == 0:
                                robot_sim.step(action_index, traj_gen.config.ts)
                            else:
                                robot_sim.step_with_ref_speed(traj_gen.config.ts, 1.0)
                            rl_ref.append(list(robot_sim.position))
                        last_rl_time = timer_rl(4, ms=True)
                        # last_rl_ref = rl_ref
                        
                        if dyn_obstacle_list:
                            # traj_gen.update_dynamic_constraints([dyn_obstacle_tmp*20])
                            traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                        other_robot_states = robot_manager.get_other_robot_states(robot_i)
                        traj_gen.update_other_robot_states(other_robot_states)

                        original_ref_traj, rl_ref_traj, extra_ref_traj = traj_gen.get_local_ref_traj(np.array(rl_ref), extra_horizon=int(traj_gen.config.N_hor*1.5))
                        filtered_ref_traj = ref_traj_filter(original_ref_traj, rl_ref_traj, decay=1.0) # decay=1 means no decay
                        other_robot_states_poly = [x for i, x in enumerate(robot_states_poly) if i != robot_i]
                        if switch.switch(traj_gen.state[:2], extra_ref_traj.tolist(), filtered_ref_traj.tolist(), geo_map.processed_obstacle_list, dyn_obstacle_list_poly+other_robot_states_poly):
                            chosen_ref_traj = filtered_ref_traj
                        else:
                            chosen_ref_traj = original_ref_traj
                        if done_list[robot_i]:
                            chosen_ref_traj = original_ref_traj
                        timer_mpc = PieceTimer()
                        try:
                            mpc_output = traj_gen.get_action(chosen_ref_traj) # MPC computes the action
                        except Exception as e:
                            done = True
                            print(f'MPC fails: {e}')
                            break
                        last_mpc_time = timer_mpc(4, ms=True)
                        
                        if mpc_output is not None:
                            action, pred_states, cost = mpc_output
                            robot_manager.set_robot_state(robot_i, traj_gen.state)
                            robot_manager.set_pred_states(robot_i, pred_states)
                        else:
                            robot_manager.set_robot_state(robot_i, traj_gen.state)
                            robot_manager.set_pred_states(robot_i, None)

                    elif decision_mode == 4:
                        env_eval.set_agent_state(traj_gen.state[:2], traj_gen.state[2], 
                                                traj_gen.last_action[0], traj_gen.last_action[1])
                        timer_rl = PieceTimer()
                        action_index, _states = td3_model.predict(obsv, deterministic=True)
                        env_eval.set_temp_obstacles([x for i, x in enumerate(robot_states_poly_obs) if i != robot_i])
                        # obsv, reward, done, info = env_eval.step(action_index)
                        ### Manual step
                        env_eval.step_obstacles()
                        env_eval.update_status(reset=False)
                        obsv = env_eval.get_observation()
                        done = env_eval.update_termination()
                        info = env_eval.get_info()

                        rl_ref = []
                        robot_sim:MobileRobot = copy.deepcopy(env_eval.agent)
                        for j in range(20):
                            if j == 0:
                                robot_sim.step(action_index, traj_gen.config.ts)
                            else:
                                robot_sim.step_with_ref_speed(traj_gen.config.ts, 1.0)
                            rl_ref.append(list(robot_sim.position))
                        last_rl_time = timer_rl(4, ms=True)
                        # last_rl_ref = rl_ref
                        
                        if dyn_obstacle_list:
                            # traj_gen.update_dynamic_constraints([dyn_obstacle_tmp*20])
                            traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)
                        other_robot_states = robot_manager.get_other_robot_states(robot_i)
                        traj_gen.update_other_robot_states(other_robot_states)
                        
                        original_ref_traj, rl_ref_traj, extra_ref_traj = traj_gen.get_local_ref_traj(np.array(rl_ref))
                        filtered_ref_traj = ref_traj_filter(original_ref_traj, rl_ref_traj, decay=1) # decay=1 means no decay
                        other_robot_states_poly = [x for i, x in enumerate(robot_states_poly) if i != robot_i]
                        if switch.switch(traj_gen.state[:2], original_ref_traj.tolist(), filtered_ref_traj.tolist(), geo_map.processed_obstacle_list+dyn_obstacle_list_poly+other_robot_states_poly):
                            chosen_ref_traj = filtered_ref_traj
                        else:
                            chosen_ref_traj = original_ref_traj
                        timer_mpc = PieceTimer()
                        try:
                            mpc_output = traj_gen.get_action(chosen_ref_traj) # MPC computes the action
                        except Exception as e:
                            done = True
                            print(f'MPC fails: {e}')
                            break
                        last_mpc_time = timer_mpc(4, ms=True)

                        if mpc_output is not None:
                            action, pred_states, cost = mpc_output
                            robot_manager.set_robot_state(robot_i, traj_gen.state)
                            robot_manager.set_pred_states(robot_i, pred_states)
                        else:
                            robot_manager.set_robot_state(robot_i, traj_gen.state)
                            robot_manager.set_pred_states(robot_i, None)

                    else:
                        raise ValueError("Invalid decision mode")
                    
                    obsv_list[robot_i] = obsv

                    if not to_plot:
                        print(f"Step {i}.")

                    if decision_mode == 0:
                        time_list.append(last_mpc_time)
                        if to_plot:
                            print(f"Step {i}.Runtime (MPC): {last_mpc_time}ms")
                    elif decision_mode == 1:
                        time_list.append(last_rl_time)
                        if to_plot:
                            print(f"Step {i}.Runtime (DDPG): {last_rl_time}ms")
                    elif decision_mode == 2:
                        time_list.append(last_rl_time)
                        if to_plot:
                            print(f"Step {i}.Runtime (TD3): {last_rl_time}ms")
                    elif decision_mode == 3:
                        time_list.append(last_mpc_time+last_rl_time)
                        if to_plot:
                            print(f"Step {i}.Runtime (Hybrid DDPG): {last_mpc_time+last_rl_time} = {last_mpc_time}+{last_rl_time}ms")
                    elif decision_mode == 4:
                        time_list.append(last_mpc_time+last_rl_time)
                        if to_plot:
                            print(f"Step {i}.Runtime (Hybrid TD3): {last_mpc_time+last_rl_time} = {last_mpc_time}+{last_rl_time}ms") 


                    if to_plot & (i%1==0): # render
                        vis_component = True if robot_i == 0 else False
                        if vis_component:
                            vis_component = False if decision_mode == 0 else True
                        env_eval.render_ax(ax_main, ax_profile, ax_obsv=ax_obsv, plot_map=plot_map, plot_obsv=plot_obsv, vis_component=vis_component,
                                           drl_ref=rl_ref, actual_ref=chosen_ref_traj, original_ref=original_ref_traj, theme_color=theme_color)
                        # add text at the right bottom
                        ax_main.text(0.99, 0.01, f"Timestep: {i}", fontsize=16, ha='right', va='bottom', transform=ax_main.transAxes)
                        plot_map = False
                        plot_obsv = False

                    if done:
                        done_list[robot_i] = True
                        print(f"Robor {robot_i} finish (Succeed: {info['success']})!")

                if to_plot:
                    plt.pause(0.01)

                    ### Legend - main
                    leg_border = Line2D([0], [0], color='black', marker='None', linestyle='-', label='Map border')
                    leg_obstacle = Line2D([0], [0], color='red', marker='None', linestyle='-', label='Obstacle')
                    leg_obstacle_pad = Line2D([0], [0], color='red', marker='None', linestyle='--', label='Obstacle (padded)')

                    leg_robot = Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=10, label='Robot')
                    leg_past_traj = Line2D([0], [0], color='blue', marker='None', linestyle='-', label='Past trajectory')
                    leg_original_ref = Line2D([0], [0], color='blue', marker='o', linestyle='-', markerfacecolor='none', label='Original reference')
                    leg_actual_ref = Line2D([0], [0], color='blue', marker='x', linestyle='-', label='Actual reference')
                    leg_rl_ref = Line2D([0], [0], color='blue', marker='s', linestyle='-', markerfacecolor='none', label='RL reference')
                    if scene_option != 0:
                        handles = [leg_border, leg_obstacle, leg_obstacle_pad, leg_robot, leg_past_traj, leg_original_ref, leg_actual_ref]
                    else:
                        handles = [leg_border, leg_robot, leg_past_traj, leg_original_ref, leg_actual_ref]
                    if decision_mode in [3, 4]:
                        handles.append(leg_rl_ref)
                    if map_only:
                        ax_main.legend(handles=handles, loc='upper left', prop={'size': 16}, ncol=3)
                        ax_main.set_xticks([])
                        ax_main.set_yticks([])
                    else:
                        ax_main.legend(handles=handles, loc='upper left')
                    # ax_main.set_title(f'Time step {i} (different colors for different robots)', fontsize=16)
                    # ax_main.set_title(f'Simulation for {len(env_eval_list)} robots (different colors for different robots)')
                    ### Legend - profile
                    leg_speed = Line2D([0], [0], color='blue', marker='None', linestyle='-', label='Speed')
                    leg_angular_speed = Line2D([0], [0], color='blue', marker='.', linestyle='dashed', label='Angular speed')
                    if ax_profile is not None:
                        ax_profile.legend(handles=[leg_speed, leg_angular_speed], loc='upper right')
                        ax_profile.set_title('Motion profile (different colors for different robots)')
                    if ax_obsv is not None:
                        ax_obsv.set_title('Observation (from one robot)')

                    # while True:
                    #     if plt.waitforbuttonpress(timeout=0.1):
                    #         break
                    if save:
                        # if doesn't exist, create the folder and file
                        if not os.path.exists(f'./src/results/{i}.png'):
                            os.makedirs(f'./src/results', exist_ok=True)
                        plt.savefig(f'./src/results/{i}.png')

                if all(done_list):
                    break

            if i == MAX_RUN_STEP - 1:
                print(f'Time out!')
                break

        if to_plot:
            plt.show()

    print(f"Average time ({prt_decision_mode[decision_mode]}): {np.mean(time_list)}ms\n")
    return time_list

if __name__ == '__main__':
    import sys
    """
    rl_index: 0 = image, 1 = ray
    decision_mode: 0 = MPC, 1 = DDPG, 2 = TD3, 3 = Hybrid DDPG, 4 = Hybrid TD3  
    """
    ### 0 - empty,    1 - single obstacle,    2 - plus-shaped obstacle, 
    ### 3 - 6 robots, 4 - Track crossing,     5 - Parallel tunnel
    scene_option = 5

    save = False
    to_plot = True
    map_only = False

    # time_list_mpc     = main(rl_index=1,    decision_mode=0,  to_plot=to_plot, map_only=map_only, scene_option=scene_option, save=save)
    # time_list_lid     = main(rl_index=1,    decision_mode=1,  to_plot=to_plot, map_only=map_only, scene_option=scene_option, save=save)
    # time_list_img     = main(rl_index=0,    decision_mode=1,  to_plot=to_plot, map_only=map_only, scene_option=scene_option, save=save)
    # time_list_hyb_lid = main(rl_index=1,    decision_mode=3,  to_plot=to_plot, map_only=map_only, scene_option=scene_option, save=save)
    time_list_hyb_img = main(rl_index=0,    decision_mode=3,  to_plot=to_plot, map_only=map_only, scene_option=scene_option, save=save)
    sys.exit()

    print(f"Average time: \nDDPG {np.mean(time_list_lid)}ms; \nMPC {np.mean(time_list_mpc)}ms; \nHYB {np.mean(time_list_hyb_lid)}ms; \n")

    fig, axes = plt.subplots(1,2)

    bin_list = np.arange(0, 150, 10)
    axes[0].hist(time_list_lid, bins=bin_list, color='r', alpha=0.5, label='DDPG')
    axes[0].hist(time_list_mpc, bins=bin_list, color='b', alpha=0.5, label='MPC')
    axes[0].hist(time_list_hyb_lid, bins=bin_list, color='g', alpha=0.5, label='HYB')
    axes[0].legend()

    axes[1].plot(time_list_lid, color='r', ls='-', marker='x', label='DDPG')
    axes[1].plot(time_list_mpc, color='b', ls='-', marker='x', label='MPC')
    axes[1].plot(time_list_hyb_lid, color='g', ls='-', marker='x', label='HYB')

    plt.show()
    input('Press enter to exit...')