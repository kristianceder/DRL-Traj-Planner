import numpy as np


class RobotManager:
    def __init__(self, num_robots: int, config_mpc):
        self._num_robots = num_robots
        self._robot_dict = {}

        self.state_dim = config_mpc.ns
        self.horizon = config_mpc.N_hor
        self.num_others = config_mpc.Nother

    def add_robot(self, robot_id: int):
        state = None
        pred_states = None
        self._robot_dict[robot_id] = [state, pred_states]

    def set_robot_state(self, robot_id: int, state: np.ndarray):
        self._robot_dict[robot_id][0] = state

    def set_pred_states(self, robot_id: int, pred_states: np.ndarray):
        self._robot_dict[robot_id][1] = pred_states

    def get_robot_state(self, robot_id: int) -> np.ndarray:
        return self._robot_dict[robot_id][0]
    
    def get_pred_states(self, robot_id: int) -> np.ndarray:
        return self._robot_dict[robot_id][1]

    def get_other_robot_states(self, ego_robot_id, default:float=-10.0) -> list:
        # other_robot_states = [default] * self.state_dim * (self.horizon+1) * self.num_others
        other_robot_states = [default] * self.state_dim * (self.horizon) * self.num_others
        idx = 0
        idx_pred = self.state_dim * self.num_others
        for id_ in list(self._robot_dict):
            if id_ != ego_robot_id:
                current_state:np.ndarray = self.get_robot_state(id_)
                pred_states:np.ndarray = self.get_pred_states(id_) # every row is a state
                other_robot_states[idx : idx+self.state_dim] = list(current_state)
                idx += self.state_dim
                if pred_states is not None:
                    # other_robot_states[idx_pred : idx_pred+self.state_dim*self.horizon] = list(pred_states.reshape(-1))
                    other_robot_states[idx_pred : idx_pred+self.state_dim*(self.horizon-1)] = list(np.array(pred_states).reshape(-1))[:-3]
                    idx_pred += self.state_dim*self.horizon
        return other_robot_states