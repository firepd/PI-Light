import re
import time
import random
from math import log, sqrt
from typing import List
import itertools
from skopt import gp_minimize
from skopt.space import Real, Integer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from env import TSCEnv
from agent.pi_light.program import Bale
from agent.pi_light.pi_light import PiLight
from agent.pi_light.utils import *


def pi_run_a_step(env: TSCEnv, n_obs: List):
    n_action = []
    for agent in env.n_agent:
        action = agent.pick_action(n_obs, None)
        n_action.append(action)
    n_next_obs, n_rew, n_done, info = env.step(n_action)
    return n_next_obs, n_rew, n_done, info, n_action[0]


def run_an_episode(env: TSCEnv, config: dict):
    n_obs = env.reset()  # Observations of n agents
    n_done = [False]
    action_sequence = []
    info = {}

    for config['current_episode_step_idx'] in itertools.count(start=0, step=config['action_interval']):
        if config['current_episode_step_idx'] >= config['num_step'] or all(n_done):
            break
        n_next_obs, n_rew, n_done, info, action = pi_run_a_step(env, n_obs)
        n_obs = n_next_obs
        action_sequence.append(str(action))

    travel_time = info['world_2_average_travel_time'][0]
    return travel_time, ''.join(action_sequence)


def get_reward(travel_time):
    assert travel_time > 20
    return 200 / travel_time


class MCTS_synthesizer:
    def __init__(self, env:TSCEnv, config):
        self.env = env
        self.n_agent = env.n_agent  # type: List[PiLight]
        self.config = config
        agent_config = self.config[self.config['cur_agent']]  # type: dict
        self.feature_list = agent_config['observation_feature_list']

        self.weight = 0.5
        self.total_episode = 0
        self.root = None
        self.library = Library()  # Save the searched two code, complexity, score
        self.optimizer = Optimizer(self.get_running_metric)
        self.visited_bank = Memory()

    def distribute(self, best=True):  # Give the synthesized program to each agent
        if best:
            inlane_code, outlane_code = self.library.query_best()
        else:
            inlane_code, outlane_code = self.library.query_top_40()

        for i in self.n_agent:
            i.inject_code(inlane_code, outlane_code)

    def init_start_programs(self):
        simple_program_pair = Bale.get_start_programs()
        self.root.evaled = True
        self.total_episode += len(simple_program_pair)
        for bale in simple_program_pair:
            self.visited_bank.check_string_duplicate(bale.output_code())
            travel_time, action_seq = self.evaluate(bale)
            self.visited_bank.check_action_duplicate(action_seq, bale.output_code())
            reward = get_reward(travel_time)
            mcts_node = MCTS_Node(self.root, bale, weight=self.weight)
            mcts_node.evaled = True
            mcts_node.update(reward)
            self.root.visits += 1
            self.root.children.append(mcts_node)
            self.library.add(bale.output_code(), 1, travel_time)

    def begin_search(self, train_episodes):
        self.root = MCTS_Node(None, None, weight=self.weight)
        self.init_start_programs()

        start = time.time()
        for i in range(1, train_episodes + 1):
            if i % 15 == 0:
                time_spend = round(time.time() - start, 2)
                print(f'time spent:{time_spend}; {i} program evaluated')

            bottom_node, metric = self.search_one()
            
            if bottom_node is not None:
                bale = bottom_node.bale  # type: Bale
                self.library.add(bale.output_code(), bale.get_complexity(), metric)

        self.library.get_pareto_frontier()
        print('total number of episode:', self.total_episode)

    def search_one(self):
        node = self.root
        epsilon = 0.1

        while node.evaled and len(node.children) > 0:
            if np.random.rand() < epsilon:
                node = node.random_select()
            else:
                node = node.select()

        if node.evaled:
            node = node.expand(self.visited_bank)
            if node is None:
                return None, 10000

        metric, action_seq = self.evaluate(node.bale)  # Evaluate and optimize parameters at the same time
        node.evaled = True

        self.visited_bank.check_action_duplicate(action_seq, node.bale.output_code())

        bottom_node = node
        reward = get_reward(metric)
        # backpropagation
        while node is not None:
            node.update(reward)
            node = node.parent

        return bottom_node, metric

    def evaluate(self, bale):
        inlane_code, outlane_code = bale.output_code()
        num_const = inlane_code.count('$') // 2 + outlane_code.count('$') // 2

        if num_const > 0:
            (inlane_code, outlane_code), opt_step = self.optimizer.optimize_const(inlane_code, outlane_code)
            bale.replace_code(inlane_code, outlane_code)
            self.total_episode += opt_step
        self.total_episode += 1
        travel_time, action_seq = self.get_running_metric(inlane_code, outlane_code)
        return travel_time, action_seq

    def get_running_metric(self, inlane_code: str, outlane_code: str):
        for i in self.n_agent:
            i.inject_code(inlane_code, outlane_code)
        travel_time, action_seq = run_an_episode(self.env, self.config)
        return travel_time, action_seq

    def check_const_meanning(self, node):
        cur_const = node.bale
        if cur_const == 0:
            return False
        a = True
        if a:
            node.parent.children.remove(node)
            return True
        else:
            return False


class MCTS_Node:
    def __init__(self, parent, bale: Bale, weight=None):
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.weight = weight
        self.bale = bale
        self.evaled = False

    def random_select(self):
        return random.sample(self.children, k=1)[0]

    def select(self):
        weights = [(c.value + self.weight * sqrt(log(self.visits) / (c.visits+0.1))) for c in self.children]
        max_idx = np.argmax(weights)
        return self.children[max_idx]

    def expand(self, visited_bank):
        possible_expands = self.bale.get_valid_expansions()
        for i in possible_expands:
            codes = i.output_code()
            if visited_bank.check_string_duplicate(codes):
                continue
            son = MCTS_Node(self, i, weight=self.weight)
            self.children.append(son)
        if len(self.children) > 0:
            child = random.sample(self.children, 1)[0]
            return child
        else:
            self.value = -1
            self.visits = 10000
            return None

    def update(self, reward):
        self.visits += 1
        self.value = max(self.value, reward)

    def __repr__(self):
        return f'MCTS node, value:{self.value}, visits:{self.visits}, evaled:{self.evaled}'


class Optimizer:
    def __init__(self, run_func):
        self.run_func = run_func
        self.condition_param_space = {'inlane_2_num_vehicle': [0, 40], 'outlane_2_num_vehicle': [0, 20], 'inlane_2_num_waiting_vehicle': [0, 40]}
        self.condition_default_value = 10
        self.weight_range = [0.1, 8]
        self.weight_default_value = 1
        self.dist_threshold_range = {'inlane_2_vehicle_dist': [5, 200], 'outlane_2_vehicle_dist': [5, 200]}
        self.threshold_default_value = {'inlane_2_vehicle_dist': 150, 'outlane_2_vehicle_dist': 10}

    def param_count(self, code:str):
        num_cond = code.count('$cond:')
        num_thresh = code.count('$thresh:')
        num_weight = code.count('$weight:')
        return num_cond, num_thresh, num_weight

    def gen_range(self, code:str):
        kind_feat_name = re.findall('\$(\w+):(\w+)\$', code)
        param_ranges = []
        default_values = []
        for kind, feat_name in kind_feat_name:
            if kind == 'cond':
                param_range = self.condition_param_space[feat_name]
                param_range = Integer(param_range[0], param_range[1], prior='uniform')
                default_value = self.condition_default_value + int(np.random.rand() * 5)
            elif kind == 'weight':
                param_range = self.weight_range
                param_range = Real(param_range[0], param_range[1], prior='uniform')
                default_value = self.weight_default_value
            elif kind == 'thresh':
                param_range = self.dist_threshold_range[feat_name]
                param_range = Integer(param_range[0], param_range[1], prior='uniform')
                default_value = self.threshold_default_value[feat_name]
            else:
                assert 0
            param_ranges.append(param_range)
            default_values.append(default_value)
        return param_ranges, default_values

    def optimize_const(self, inlane_code, outlane_code):
        in_param_ranges, in_default_values = self.gen_range(inlane_code)
        out_param_ranges, out_default_values = self.gen_range(outlane_code)
        self.num_in_param = len(in_default_values)
        self.in_parts, self.out_parts = self.decompose(inlane_code, outlane_code)
        param_ranges = in_param_ranges + out_param_ranges
        default_values = in_default_values + out_default_values

        opt_step = self.get_opt_step(len(default_values))
        start = time.time()
        result = gp_minimize(self._evaluate, dimensions=param_ranges, n_calls=opt_step, n_initial_points=3, x0=default_values)
        # print('optimize_const time cost:', time.time() - start)
        return self.assemble(result.x), opt_step

    def _evaluate(self, params):
        in_full_code, out_full_code = self.assemble(params)
        travel_time, action_seq = self.run_func(in_full_code, out_full_code)
        return travel_time

    def get_opt_step(self, num_const):
        setting = {1:4, 2:7, 3: 10}
        return setting.get(num_const, 12)

    def decompose(self, in_code: str, out_code: str):
        in_parts = re.split('\$\w+:\w+\$', in_code)
        out_parts = re.split('\$\w+:\w+\$', out_code)
        return in_parts, out_parts

    def assemble(self, values):
        num_in_param = self.num_in_param
        in_param_values = values[:num_in_param]
        out_param_values = values[num_in_param:]
        in_full_code = concatenate(self.in_parts, in_param_values)
        out_full_code = concatenate(self.out_parts, out_param_values)
        return in_full_code, out_full_code


def concatenate(code_parts, values):
    result = ''
    for i in range(min(len(code_parts), len(values))):
        result += code_parts[i] + str(values[i])

    if len(code_parts) > len(values):
        result += ''.join(x for x in code_parts[len(values):])
    elif len(values) > len(code_parts):
        result += ''.join(str(x) for x in values[len(code_parts):])
    return result


