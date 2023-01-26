from tqdm import tqdm
import numpy as np
from src.batch_bandits.batch_k_ucb import BatchKernelUCB
from src.batch_bandits.sbpe import SBPE
from src.batch_bandits.kernels import Polynomial, Exponential, Linear, Gaussian

from utils.dataset import get_dataset_by_name
from utils.utils import LossHistory, online_evaluation, start_experiment, get_logging_data, update_past_data

import os
import sys
import time
import os
from datetime import date
today = date.today()

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)

class Environment:

    def __init__(self, dataset, n_logging_samples):
        self.dataset = dataset
        self.n_logging_samples = n_logging_samples

    def sample_data(self, n):
        return self.dataset.sample_data(n_samples=n, index=0)

    def sample_reward(self, actions, labels):
        actions = np.squeeze(actions)
        return - self.dataset.get_losses_from_actions(labels, actions)

    def get_anchor_points(self):
        return np.arange(-5, 5, 0.5)

    def get_logging_data(self):
        actions, contexts, losses, _ = self.dataset.get_logging_data(self.n_logging_samples)
        return actions, contexts, -losses


def instantiate_metrics():
    return {
        'online_loss': [],
        'cumulated_loss': [],
    }


def get_agent(settings, kernel):
    if settings['agent'] == 'BKUCB':
        return BatchKernelUCB(settings, kernel)
    else:
        return SBPE(settings, kernel)

def get_kernel(settings):
    if settings['kernel'] == 'polynomial':
        return Polynomial(settings)
    elif settings['kernel'] == 'linear':
        return Linear(settings)
    elif settings['kernel'] == 'gaussian':
        return Gaussian(settings)
    else:
        return Exponential(settings)

def save_result(dataset_name, random_seed, settings, rollout, online_loss, total_time):
    task_name = 'algo:{}'.format(settings['agent'])
    task_name += '|{}:{}'.format('lambda', settings['lambda'])
    task_name += '|{}:{}'.format('rd', random_seed)
    task_name += '|{}:{}'.format('env', dataset_name)
    task_name += '|{}:{}'.format('rollout', rollout)
    metrics_information = 'online_loss:{}'.format(online_loss)
    metrics_information += '|total_time:{}'.format(total_time)

    result = '{} {}\n'.format(task_name, metrics_information)

    results_dir = 'results/{}'.format(today.strftime("%d-%m-%Y"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fname = os.path.join(results_dir, 'metrics.txt')

    with open(fname, 'a') as file:
        file.write(result)


def batch_bandit_experiment(random_seed, dataset_name, settings):
    dataset = get_dataset_by_name(dataset_name, random_seed)

    start_experiment(random_seed, dataset, settings['agent'])

    # Model setting
    env = Environment(dataset, settings['n_0'])
    kernel = get_kernel(settings)
    agent = get_agent(settings, kernel)
    agent.instantiate(env)
    metrics = instantiate_metrics()
    batch_size = settings['n_0']

    t0 = time.time()

    for step in tqdm(range(settings['M'])):
        # choose a random context.
        batch_size *= 2
        contexts, labels = env.sample_data(n=batch_size)
        # iterate learning algorithm for 1 round.
        actions = agent.sample_actions(contexts)
        rewards = env.sample_reward(actions, labels)

        agent.update_agent(contexts, actions, rewards)
        t = time.time() - t0

        contexts, potentials = dataset.test_data
        actions = agent.sample_actions(contexts)
        rewards = env.sample_reward(actions, labels)
        online_loss = -np.mean(rewards)
        metrics['online_loss'].append(online_loss)
        print('Rollout {}, Online reward: {}'.format(step, -metrics['online_loss'][-1]))
        save_result(dataset_name, random_seed, settings, step, online_loss, t)

    batch_online_losses = np.array([online_loss._value for online_loss in metrics['online_loss']])
    return np.expand_dims(batch_online_losses, axis=0)


