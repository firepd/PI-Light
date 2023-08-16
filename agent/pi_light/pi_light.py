from env.intersection import Intersection
from env.TSC_env import TSCEnv
import numpy as np


class PiLight:
    def __init__(self, config, env: TSCEnv, idx):
        self.config = config
        self.env = env  # type: TSCEnv
        self.idx = idx

        self.inter = env.n_intersection[idx]  # type: Intersection
        self.action_space = env.n_action_space[idx]
        self.num_phase = self.action_space.n
        self.current_phase = 0

        agent_config = self.config[self.config['cur_agent']]  # type: dict
        self.feature_list = agent_config['observation_feature_list']

        self.inlane_code = ''
        self.outlane_code = ''

    def inject_code(self, inlane_code, outlane_code):
        self.inlane_code = inlane_code
        self.outlane_code = outlane_code

    def reset(self):
        pass

    def pick_action(self, n_obs, on_training):
        obs = n_obs[self.idx]
        assert len(obs) == len(self.feature_list)

        num_move = len(self.inter.n_roadlink)
        move_values = np.zeros(num_move)
        for move_id in range(num_move):
            move_values[move_id] = self._get_value_for_move2(obs, move_id)

        phase_values = self._aggregate_for_each_phase(move_values)
        action = phase_values.argmax()
        self.current_phase = action
        return action

    def _get_value_for_move(self, obs, move_id):
        inlane_2_num_vehicle, outlane_2_num_vehicle, inlane_2_num_waiting_vehicle, inlane_2_vehicle_dist, outlane_2_vehicle_dist = obs
        value = [0]
        n_startlane_id = self.inter.n_roadlink[move_id].n_startlane_id
        n_endlane_id = self.inter.n_roadlink[move_id].n_endlane_id
        for start_lane_name in n_startlane_id:
            index = self.inter.n_in_lane_id.index(start_lane_name)
            exec(self.inlane_code)

        for end_lane_name in n_endlane_id:
            index = self.inter.n_out_lane_id.index(end_lane_name)
            exec(self.outlane_code)
        return value[0]

    def _get_value_for_move2(self, obs, move_id):
        inlane_2_num_vehicle, outlane_2_num_vehicle, inlane_2_num_waiting_vehicle, inlane_2_vehicle_dist, outlane_2_vehicle_dist = obs
        value = [0]
        n_lanelink_id = self.inter.n_roadlink[move_id].n_lanelink_id
        for lane_link in n_lanelink_id:
            start_lane_name, end_lane_name = lane_link[0], lane_link[1]
            index = self.inter.n_in_lane_id.index(start_lane_name)
            exec(self.inlane_code)
            index = self.inter.n_out_lane_id.index(end_lane_name)
            exec(self.outlane_code)

        return value[0]

    def _aggregate_for_each_phase(self, move_values):
        phase_values = np.zeros(self.num_phase)
        for phase_id in range(self.num_phase):
            n_roadlink_idx = self.inter.n_phase[phase_id].n_available_roadlink_idx
            phase_values[phase_id] = move_values[n_roadlink_idx].sum()
        return phase_values

