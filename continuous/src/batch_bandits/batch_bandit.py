from tqdm import tqdm
import numpy as np
from src.batch_bandits.batch_k_ucb import BatchKernelUCB
from src.batch_bandits.sbpe import SBPE
from src.batch_bandits.kernels import Polynomial, Exponential

from utils.dataset import get_dataset_by_name
from utils.utils import LossHistory, online_evaluation, start_experiment, get_logging_data, update_past_data

class Environment:

    def __init__(self, dataset, n_logging_samples):
        self.dataset = dataset
        self.n_logging_samples = n_logging_samples

    def sample_data(self, n):
        return self.dataset.sample_data(n_samples=n)

    def sample_reward(self, actions, labels):
        actions = np.squeeze(actions)
        return - self.dataset.get_losses_from_actions(labels, actions)

    def get_anchor_points(self):
        return np.arange(-5, 5, 0.5)

    def get_logging_data(self):
        actions, contexts, losses, _, _ = self.dataset.sample_logged_data(self.n_logging_samples)
        return actions, contexts, -losses


def instantiate_metrics():
    return {
        'online_loss': [],
        'cumulated_loss': [],
    }



def get_agent(settings, kernel):
    if settings['agent'] == 'Batch-KUCB':
        return BatchKernelUCB(settings, kernel)
    else:
        return SBPE(settings, kernel)

def get_kernel(settings):
    if settings['kernel'] == 'polynomial':
        return Polynomial(settings)
    else:
        return Exponential(settings)


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

    for step in tqdm(range(settings['M'])):
        # choose a random context.
        batch_size *= 2
        contexts, labels = env.sample_data(n=batch_size)
        # iterate learning algorithm for 1 round.
        actions = agent.sample_actions(contexts)
        rewards = env.sample_reward(actions, labels)

        agent.update_agent(contexts, actions, rewards)

        metrics['online_loss'].append(-np.mean(agent.past_rewards[-batch_size:]))
        metrics['cumulated_loss'].append(np.sum(-agent.past_rewards[1:]))
        print('Rollout {}, Online reward: {}'.format(step, -metrics['online_loss'][-1]))

    batch_online_losses = np.array([online_loss._value for online_loss in metrics['online_loss']])
    batch_cumulated_losses = np.array([cumulated_loss._value for cumulated_loss in metrics['cumulated_loss']])
    return np.expand_dims(batch_online_losses, axis=0), np.expand_dims(batch_cumulated_losses, axis=0)


