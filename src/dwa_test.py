"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import gym
import copy
import shapely
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

from pkg_ddpg_td3.utils.map import generate_map_eval
from pkg_ddpg_td3.utils.map_simple import *
from pkg_ddpg_td3.inference_model import inference_model
from pkg_ddpg_td3.utils.map_eval import generate_eval_map111

def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.5  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 0.5  # [rad/s]
        self.max_accel = 1  # [m/ss]
        self.max_delta_yaw_rate = 3  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1  # [rad/s]
        self.dt = 0.2  # [s] Time tick for motion prediction
        self.predict_time = 4.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.5  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = np.array([[-1, -1],
                            [0, 2],
                            [4.0, 2.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [5.0, 9.0],
                            [8.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 10.0],
                            [9.0, 11.0],
                            [12.0, 13.0],
                            [12.0, 12.0],
                            [15.0, 15.0],
                            [13.0, 13.0]
                            ])



config = Config()


def motion(x, u, dt):
    """
    motion model
    """
    # ours state: (x, y, theta, v, w)
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    # u = [a(m/s^2),w.dot]
    x[2] += u[1] * dt   # angle
    x[3] = u[0]         # velocity
    x[4] = u[1]         # yaw rate

    x[0] += x[3] * math.cos(x[2]) * dt # x-pos
    x[1] += x[3] * math.sin(x[2]) * dt # y-pos
    
    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """
    # ox = ob[:, 0]
    # oy = ob[:, 1]
    # dx = trajectory[:, 0] - ox[:, None]
    # dy = trajectory[:, 1] - oy[:, None]
    # r = np.hypot(dx, dy)
    r_min = float("Inf")
    for i in range(len(trajectory)-1):
        for o in ob.obstacles:
            r = o._padded_polygon.distance(LineString([tuple(trajectory[i,0:2]),tuple(trajectory[i+1,0:2])]))
            if r <= 0:
                print("negative!")
                return float("Inf")
            elif r < r_min:
                r_min = r
            
            # if o.collides(MobileRobot(trajectory[i,:])):
                
            
            # dx = nearest_points(Point(trajectory[0,0],trajectory[0,1]),ob.obstacles[0]._padded_polygon)[0].x
    # if np.array(r <= config.robot_radius).any():
    #         return 
    return 1.0 / r_min  # OK


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def get_rl_ref(action,agent):
    rl_ref = []
    robot_sim:MobileRobot = copy.deepcopy(agent)
    robot_sim:MobileRobot
    for j in range(20):
        if j == 0:
            robot_sim.step(action, 0.2)
        else:
            robot_sim.step_with_ref_speed(0.2, 1.0)
        rl_ref.append(list(robot_sim.position))
    return rl_ref

def run():
    
    # Select the path where the model should be stored
    # path = './Model/testing/variant-0/run1'
    # path = './Model/testing/variant-6'
    path = './Model/ddpg/ray'
    env_eval = gym.make('TrajectoryPlannerEnvironmentRaysReward1-v0', generate_map=generate_eval_map111)

    model = inference_model(path,env_eval)

    while True:
        obs = env_eval.reset()
        x = np.copy(env_eval.agent.state)
        x_old = np.copy(env_eval.agent.state)
        goal = env_eval.goal.position
        trajectory = np.array(x)
        for i in range(0, 1000):
            u, predicted_trajectory = dwa_control(x, config, goal, env_eval)
            a = (u-x[3:5])/0.2


            action = model.get_action(obs)
            rl_ref = get_rl_ref(action,env_eval.agent)
            dwa_ref = predicted_trajectory[:,0:2].tolist()
            obs, reward, done, info = env_eval.step(action)
            x = np.copy(env_eval.agent.state)
            if i % 1 == 0: # Only render every third frame for performance (matplotlib is slow)
                # vec_env.render("human")
                env_eval.render(dqn_ref=dwa_ref)
            if done:
                break
    
if __name__ == "__main__":
    run()


# obstacleData[0].collides(amrData)