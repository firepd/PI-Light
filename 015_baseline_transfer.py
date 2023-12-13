from env import TSCEnv
from utilities.utils import set_seed, get_agent, get_config, set_thread, set_logger, release_logger, make_dir, copy_model_params
from utilities.snippets import run_an_episode
import numpy as np
import multiprocessing
import argparse
import os
from agent.drl_based.base_drl import BaseDRL


parser = argparse.ArgumentParser(description='ATSC baselines')
parser.add_argument('--model', type=str, default='CoLight', help='[CoLight, EcoLight, FRAP, MPLight]')
parser.add_argument('--dataset', type=str, default='Hangzhou2', help='[Hangzhou1, Hangzhou2, Hangzhou3, Manhattan, Atlanta, Jinan, LosAngeles]')
parser.add_argument('--target', type=str, default='Manhattan', help='[Hangzhou1, Hangzhou2, Hangzhou3, Manhattan, Atlanta, Jinan, LosAngeles]')
args = parser.parse_args()
DEBUG = False
cur_agent = args.model
transfer_target = args.target
data_name = args.dataset


def run_an_experiment(inter_name, flow_idx, seed):
    num_step = {'Atlanta': 900, 'Hangzhou1': 3600, 'Hangzhou2': 3600, 'Hangzhou3': 3600, 'Jinan': 3600, 'LosAngeles': 1800}
    config = get_config()
    config.update({
        'inter_name': inter_name,
        'seed': seed,
        'flow_idx': flow_idx,  # 0~10
        "saveReplay": False,
        'save_result': not DEBUG,
        'dir': 'data/{}/'.format(inter_name),
        'flowFile': 'flow_{}.json'.format(flow_idx),
        'cur_agent': cur_agent,
        'render': False,
        'num_step': num_step[inter_name] if inter_name in num_step.keys() else 3600,
    })
    set_seed(seed)
    set_thread()
    set_logger(config)

    env = TSCEnv(config)
    env.n_agent = []
    for idx in range(env.n):
        agent = get_agent(cur_agent)(config, env, idx)
        env.n_agent.append(agent)

    if isinstance(env.n_agent[0], BaseDRL):
        epsilon = config[cur_agent]['epsilon']
        for config['current_episode_idx'] in range(1, 64 + 1):
            config[cur_agent]['epsilon'] = max(0, epsilon - epsilon / 55 * config['current_episode_idx'])
            info = run_an_episode(env, config, on_training=True, store_experience=True, learn=True)
            config['logger'].info('[{} On Training] Inter: {}; Flow: {}; Episode: {}; ATT: {:.2f}; AQL: {:.2f}; AD: {:.2f}; Throughput: {:.2f}'.format(
                cur_agent, inter_name, flow_idx, config['current_episode_idx'],
                info['world_2_average_travel_time'][0],
                info['world_2_average_queue_length'][0],
                info['world_2_average_delay'][0],
                info['world_2_average_throughput'][0],
            ))

    config.update({
        'inter_name': transfer_target,
        'flow_idx': flow_idx,  # 0~10
        'save_result': False,
        'dir': 'data/{}/'.format(transfer_target),
        'flowFile': 'flow_{}.json'.format(flow_idx),
        'num_step': num_step[transfer_target] if transfer_target in num_step.keys() else 3600,
    })

    target_env = TSCEnv(config)
    target_env.n_agent = []
    for idx in range(target_env.n):
        agent = get_agent(cur_agent)(config, target_env, idx)
        copy_model_params(env.n_agent[0].network_local, agent.network_local)
        target_env.n_agent.append(agent)

    info = run_an_episode(target_env, config, on_training=False, store_experience=False, learn=False)
    config['logger'].info('[{} On Generalization] Inter: {}; Flow: {}; Episode: {}; ATT: {:.2f}; AQL: {:.2f}; AD: {:.2f}; Throughput: {:.2f}'.format(
        cur_agent, transfer_target, flow_idx, config['current_episode_idx'],
        info['world_2_average_travel_time'][0],
        info['world_2_average_queue_length'][0],
        info['world_2_average_delay'][0],
        info['world_2_average_throughput'][0],
    ))

    release_logger(config)
    return info['world_2_average_travel_time'][0], \
           info['world_2_average_queue_length'][0], \
           info['world_2_average_delay'][0], \
           info['world_2_average_throughput'][0]


if __name__ == '__main__':
    if not DEBUG:
        make_dir('log/{}/{}/'.format(data_name, cur_agent))
    parallel = not DEBUG
    total_run = 10
    if os.cpu_count() == 16:
        num_concurrent_p = 10
    else:
        num_concurrent_p = total_run

    metrics = {
        'travel_time': [None for _ in range(total_run)],
        'queue_length': [None for _ in range(total_run)],
        'delay': [None for _ in range(total_run)],
        'throughput': [None for _ in range(total_run)]
    }
    seed_list = [992832, 284765, 905873, 776383, 198876, 192223, 223341, 182228, 885746, 992817]

    if parallel:
        with multiprocessing.Pool(processes=num_concurrent_p) as pool:
            n_return_value = pool.starmap(run_an_experiment, [(data_name, f_idx, seed_list[f_idx]) for f_idx in range(total_run)])
            for f_idx, return_value in enumerate(n_return_value):
                metrics['travel_time'][f_idx] = return_value[0]
                metrics['queue_length'][f_idx] = return_value[1]
                metrics['delay'][f_idx] = return_value[2]
                metrics['throughput'][f_idx] = return_value[3]
    else:
        for f_idx in range(0, total_run):
            return_value = run_an_experiment(inter_name=data_name, flow_idx=f_idx, seed=seed_list[f_idx])
            metrics['travel_time'][f_idx] = return_value[0]
            metrics['queue_length'][f_idx] = return_value[1]
            metrics['delay'][f_idx] = return_value[2]
            metrics['throughput'][f_idx] = return_value[3]

    print(f'dataset:{data_name}, method:{cur_agent}')
    print('tt: {:.2f}±{:.2f}; length: {:.2f}±{:.2f}; delay: {:.2f}±{:.2f}; through: {:.2f}±{:.2f}'.format(
        np.mean(metrics['travel_time']), np.std(metrics['travel_time']),
        np.mean(metrics['queue_length']), np.std(metrics['queue_length']),
        np.mean(metrics['delay']), np.std(metrics['delay']),
        np.mean(metrics['throughput']), np.std(metrics['throughput'])
    ))

    if not DEBUG:
        with open('log/{}/{}/summary.txt'.format(data_name, cur_agent), 'a') as fout:
            fout.write('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
                np.mean(metrics['travel_time']), np.std(metrics['travel_time']),
                np.mean(metrics['queue_length']), np.std(metrics['queue_length']),
                np.mean(metrics['delay']), np.std(metrics['delay']),
                np.mean(metrics['throughput']), np.std(metrics['throughput'])
            ))
