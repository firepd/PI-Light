import numpy as np
import itertools


def run_a_step(env, policy, n_obs):
    rollout = []
    n_action = []
    for agent in policy:
        action = agent.pick_action(n_obs, on_training=False)
        rollout.append((n_obs[agent.idx], action))
        n_action.append(action)
    n_next_obs, n_rew, n_done, info = env.step(n_action)
    return n_next_obs, n_rew, n_done, info, rollout


def run_an_episode(env, config, policy):
    n_obs = env.reset()
    n_done = [False]
    info = {}
    rollouts = []
    for config['current_episode_step_idx'] in itertools.count(start=0, step=config['action_interval']):
        if config['current_episode_step_idx'] >= config['num_step'] or all(n_done):
            break
        n_next_obs, n_rew, n_done, info, rollout = run_a_step(env, policy, n_obs)
        n_obs = n_next_obs
        rollouts.extend(rollout)
    return info, rollouts


def get_rollouts(env, config, policy, n_batch_rollouts):
    rollouts = []  # [(obs, act, rew), ...]
    for i in range(n_batch_rollouts):
        rollouts.extend(run_an_episode(env, config, policy)[1])
    return rollouts


def _sample(obss, acts, qs, max_pts, is_reweight):
    # Step 1: Compute probabilities
    ps = np.max(qs, axis=1) - np.min(qs, axis=1)
    ps = ps / np.sum(ps)

    # Step 2: Sample points
    if is_reweight:
        # According to p(s)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), p=ps)
    else:
        # Uniformly (without replacement)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), replace=False)

    # Step 3: Obtain sampled indices
    return obss[idx], acts[idx]


def to_numpy(obs: list):
    obs = [i.reshape(-1).cpu().numpy() for i in obs]
    return np.concatenate(obs, axis=0)


class TransformerPolicy:
    def __init__(self, policy, idx):
        self.policy = policy
        self.idx = idx

    def pick_action(self, n_obs, on_training):
        obs = n_obs[self.idx]
        obs = to_numpy(obs).reshape(1, -1)
        return self.policy.pick_action(obs)[0]


def identify_best_policy(env, config, policies):
    print(f'Initial policy count: {len(policies)}')
    # cut policies by half on each iteration
    while len(policies) > 1:
        # Step 1: Sort policies by current estimated reward
        policies = sorted(policies, key=lambda entry: entry[1])
        # Step 2: Prune second half of policies
        n_policies = int((len(policies) + 1)/2)
        print(f'Current policy count: {n_policies}')

        # Step 3: build new policies
        new_policies = []
        for i in range(n_policies):
            policy, rew = policies[i]
            wrapped_policy = [TransformerPolicy(policy, idx) for idx in range(env.n)]
            info = run_an_episode(env, config, wrapped_policy)[0]
            travel_time = info['world_2_average_travel_time'][0]
            new_policies.append((policy, travel_time))

        policies = new_policies

    assert len(policies) == 1
    policy, rew = policies[0]
    wrapped_policy = [TransformerPolicy(policy, idx) for idx in range(env.n)]
    info = run_an_episode(env, config, wrapped_policy)[0]
    travel_time = info['world_2_average_travel_time'][0]
    print('best travel_time:', travel_time)
    return policies[0][0], info


# see https://arxiv.org/abs/1805.08328 for detail
def train_dagger(env, config, teacher, student):
    n_batch_rollouts = 1
    max_samples = 200000
    max_iters = 50
    train_frac = 0.8
    is_reweight = True

    # Step 0: Setup
    obss, acts, qs = [], [], []
    students = []
    wrapped_student = [TransformerPolicy(student, idx) for idx in range(env.n)]

    # Step 1: Generate some supervised traces into the buffer
    trace = get_rollouts(env, config, teacher, n_batch_rollouts)
    obss.extend((to_numpy(obs) for obs, _ in trace))
    acts.extend((act for _, act in trace))
    qs.extend(teacher[0].predict_q([obs for obs, _ in trace]))

    # Step 2: Dagger outer loop
    for i in range(max_iters):
        print('Iteration {}/{}'.format(i, max_iters))

        # Step 2a: Train from a random subset of aggregated data
        cur_obss, cur_acts = _sample(np.array(obss), np.array(acts), np.array(qs), max_samples, is_reweight)
        student.train(cur_obss, cur_acts, train_frac)

        # Step 2b: Generate trace using student
        student_trace = get_rollouts(env, config, wrapped_student, n_batch_rollouts)
        student_obss = [obs for obs, _ in student_trace]

        # Step 2c: Query the oracle for supervision
        teacher_qs = teacher[0].predict_q(student_obss)
        teacher_acts = [qs.argmax() for qs in teacher_qs]

        # Step 2d: Add the augmented state-action pairs back to aggregate
        obss.extend((to_numpy(obs) for obs in student_obss))
        acts.extend(teacher_acts)
        qs.extend(teacher_qs)

        # Step 2e: Estimate the reward
        info = run_an_episode(env, config, wrapped_student)[0]
        travel_time = info['world_2_average_travel_time'][0]
        print('Student metric: {}'.format(travel_time))

        students.append((student.clone(), travel_time))

    best_student, info = identify_best_policy(env, config, students)
    return best_student, info


def test_viper(env, config, student):
    wrapped_policy = [TransformerPolicy(student, idx) for idx in range(env.n)]
    info = run_an_episode(env, config, wrapped_policy)[0]
    return info

