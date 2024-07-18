
import math
import copy
from collections import defaultdict

import torch
from shapely import LineString
from torchrl.envs import step_mdp

class SimpleController:
    def __init__(self, ref_speed):
        self.ref_speed = ref_speed
        self.waypoints = None
        self.prev_angle_diff = 0

    def reset(self, path: LineString):
        self.waypoints = copy.deepcopy([x for x in path.coords])

    def calculate_control(self, current_position, current_heading):
        if self.waypoints is None:
            print('Please call reset with a path first')

        if not self.waypoints:
            return torch.tensor([0, 0], dtype=torch.float32)  # No more waypoints to follow
        
        # Current position and heading
        x, y = current_position
        theta = current_heading
        
        # Next waypoint
        waypoint = self.waypoints[0]
        target_x, target_y = waypoint
        
        # Calculate distance and angle to the next waypoint
        dx = target_x - x
        dy = target_y - y
        distance = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx)
        
        # Calculate angle difference
        angle_diff = angle_to_target - theta
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))  # Normalize angle difference
        
        # Proportional control gains
        k_linear = 1.0
        k_angular = 2.0
        k_derivative = 1.5  # Derivative gain
        
        # Linear velocity (proportional to distance)
        v = k_linear * min(distance, self.ref_speed)
        derivative = angle_diff - self.prev_angle_diff
        
        # Angular velocity (proportional to angle difference)
        omega = k_angular * angle_diff + k_derivative * derivative
        
        # If close to the waypoint, move to the next one
        if distance < 1.5:
            self.waypoints.pop(0)

        self.prev_angle_diff = angle_diff
        
        return torch.tensor([v, omega], dtype=torch.float32)
    

def rollout(env, model, config, n_steps=2_000, do_render=False):
    state = env.reset()
    model.reset(env.unwrapped.path)
    steps = 0
    ep_rwd = torch.zeros(1).to(config.device)

    data = defaultdict(list)

    for i in range(n_steps):
        action = state.copy()
        c_action = model.calculate_control(env.unwrapped.agent.position, env.unwrapped.agent.angle)
        action['action'] = c_action
        next_state = env.step(action)

        data['obs'].append(state['observation'])
        data['action'].append(c_action)

        steps += 1
        ep_rwd += next_state['next']['reward']

        # Only render every third frame for performance (matplotlib is slow)
        if i % 3 == 0 and i > 0 and do_render:
            env.render()

        if next_state['next']['done'] or steps > config.sac.max_eps_steps:
            print(f'Episode reward {ep_rwd.item():.2f}')
            state = env.reset()
            model.reset(env.unwrapped.path)
            steps = 0
            ep_rwd = torch.zeros(1).to(config.device)
        else:
            state = step_mdp(next_state)

    return data
