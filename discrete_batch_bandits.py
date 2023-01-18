import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

from dataset_utils import load_dataset

from baselines_skylines import make_baselines_skylines
from crm_dataset import CRMDataset

import time
import os
from tqdm import tqdm

def rollout_indices(data_size, rollout_scheme, n_rollouts, min_samples=100):
    if rollout_scheme == 'doubling':
        samples = [data_size]
        for _ in range(n_rollouts):
            samples += [int(samples[-1] / 2)]
        samples = sorted([_ for _ in samples if _ > min_samples])
        return samples
    elif rollout_scheme == 'linear':
        batch_size = int(data_size/n_rollouts)
        return list(range(0, data_size, batch_size))
    else:
        raise ValueError(rollout_scheme)


from itertools import chain, repeat, islice


def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))


def pad_on_right(iterable, size, padding=None):
    iterable.reverse()
    return islice(pad_infinite(iterable, padding), size)


def pad(iterable, size, padding=0):
    l = list(pad_on_right(iterable, size, padding))
    l.reverse()
    return l


class Environment:

    def __init__(self, crm_dataset, k):
        self.crm_dataset = crm_dataset
        self.k = k

    def sample_data(self, n):
        return self.dataset.sample_data(n_samples=n, index=0)

    def sample_reward(self, actions, y):
        n = actions.shape[0]
        actions = np.array([pad(self.action_to_multilabel(int(elem)), self.k) for elem in actions])
        rewards = (1 - np.logical_xor(actions, y)).sum(axis=1).reshape((n, 1))
        rewards = rewards / self.k
        return rewards

    def get_anchor_points(self):
        return np.arange(0, 2 ** self.k)

    @staticmethod
    def action_to_multilabel(action):
        binary = bin(action).replace("0b", "")
        return [int(elem) for elem in [*binary]]

    @staticmethod
    def multilabel_to_action(multilabel):
        string = ''.join([str(elem) for elem in multilabel])
        return int(string, 2)

    def get_logging_data(self):
        actions = self.crm_dataset.actions_np
        actions = np.array([self.multilabel_to_action(elem) for elem in actions])
        rewards = self.crm_dataset.rewards_np
        contexts = self.crm_dataset.features_np
        return actions, contexts, rewards


import jax
import jax.numpy as jnp
import jaxopt
import jax.scipy as jsp
from jax import grad, jacfwd, jacrev


def hessian(f):
    return jacfwd(jacrev(f))


@jax.jit
def inverse_gram_matrix(K):
    return jnp.linalg.inv(K)


class BatchKernelUCB:

    def __init__(self, settings, kernel):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self.rng = np.random.RandomState(123)
        self.reg_lambda = settings['lambda']
        self.kernel = kernel
        self.settings = settings
        self.name = 'BKUCB'
        self.beta_t = 0.1

    #         self.instantiated = False

    def get_story_data(self):
        return self.past_states, self.past_rewards

    def set_gram_matrix(self):
        K = self.kernel.gram_matrix(self.past_states)
        K += self.reg_lambda * jnp.eye(K.shape[0])
        self.K_matrix_inverse = inverse_gram_matrix(K)

    def instantiate(self, env):
        self.action_anchors = env.get_anchor_points()
        actions, contexts, rewards = env.get_logging_data()
        states = self.get_states(contexts, actions)
        self.past_states = jnp.array(states)
        self.past_rewards = jnp.array(rewards)
        self.set_gram_matrix()

    def get_upper_confidence_bound(self, states, K_matrix_inverse, S, past_rewards):
        K_S_s = self.kernel.evaluate(S, states)
        mean = jnp.dot(K_S_s.T, jnp.dot(K_matrix_inverse, past_rewards))
        K_ss = self.kernel.evaluate(states, states)
        std = jnp.diag((1 / self.reg_lambda) * (K_ss - jnp.dot(K_S_s.T, jnp.dot(K_matrix_inverse, K_S_s))))
        ucb = jnp.squeeze(mean) + self.beta_t * jnp.sqrt(std)
        return ucb

    def sample_actions(self, contexts):
        S, rewards = self.get_story_data()
        args = self.K_matrix_inverse, S, rewards
        actions = self.discrete_inference(contexts, args)
        return actions

    def get_states(self, contexts, actions):
        batch_size = contexts.shape[0]
        contexts, actions = contexts.reshape((batch_size, -1)), actions.reshape((batch_size, 1))
        return jnp.concatenate([contexts, actions], axis=1)

    def get_ucb_actions(self, contexts, grid, args):
        return jnp.transpose(jnp.array(
            [self.get_upper_confidence_bound(self.get_states(contexts, a * np.ones((contexts.shape[0]))), *args) for a
             in grid]))

    def set_beta_t(self):
        t = self.past_states.shape[0]
        self.beta_t = 1 / np.sqrt(t)

    def discrete_inference(self, contexts, args):
        grid = self.action_anchors
        self.set_beta_t()
        ucb_all_actions = self.get_ucb_actions(contexts, grid, args)
        idx = jnp.argmax(ucb_all_actions, axis=1)
        grid = jnp.array(grid)
        return jnp.array([grid[idx]]).reshape(contexts.shape[0], 1)

    def update_data_pool(self, contexts, actions, rewards):
        states = self.get_states(contexts, actions)

        self.past_states = jnp.concatenate([self.past_states, states])
        self.past_rewards = jnp.concatenate([self.past_rewards, rewards])

    def update_agent(self, contexts, actions, rewards):
        self.update_data_pool(contexts, actions, rewards)
        self.set_gram_matrix()


class SBPE:

    def __init__(self, settings, kernel):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self.rng = np.random.RandomState(123)
        self.reg_lambda = settings['lambda']
        self.kernel = kernel
        self.settings = settings
        self.name = 'SBPE'

    def get_story_data(self):
        return self.past_states, self.past_rewards

    def set_gram_matrix(self):
        K = self.kernel.gram_matrix(self.past_states)
        self.K_matrix_inverse = inverse_gram_matrix(K)

    def instantiate(self, env):
        self.action_anchors = env.get_anchor_points()
        actions, contexts, rewards = env.get_logging_data()
        states = self.get_states(contexts, actions)
        self.past_states = jnp.array(states)
        self.past_rewards = jnp.array(rewards)
        self.set_gram_matrix()

    def pure_exploitations(self, states, K_matrix_inverse, S, past_rewards):
        K_S_s = self.kernel.evaluate(S, states)
        return jnp.squeeze(jnp.dot(K_S_s.T, jnp.dot(K_matrix_inverse, past_rewards)))

    def sample_actions(self, contexts):
        S, rewards = self.get_story_data()
        args = self.K_matrix_inverse, S, rewards
        actions = self.discrete_inference(contexts, args)
        return actions

    def get_states(self, contexts, actions):
        batch_size = contexts.shape[0]
        contexts, actions = contexts.reshape((batch_size, -1)), actions.reshape((batch_size, 1))
        return jnp.concatenate([contexts, actions], axis=1)

    def get_exploitations(self, contexts, grid, args):
        return jnp.transpose(jnp.array(
            [self.pure_exploitations(self.get_states(contexts, a * np.ones((contexts.shape[0]))), *args) for a in
             grid]))

    def discrete_inference(self, contexts, args):
        grid = self.action_anchors
        exploitations_all_actions = self.get_exploitations(contexts, grid, args)
        idx = jnp.argmax(exploitations_all_actions, axis=1)
        grid = jnp.array(grid)
        return jnp.array([grid[idx]]).reshape(contexts.shape[0], 1)

    def update_data_pool(self, contexts, actions, rewards):
        states = self.get_states(contexts, actions)
        self.past_states = jnp.concatenate([self.past_states, states])
        self.past_rewards = jnp.concatenate([self.past_rewards, rewards])

    def update_agent(self, contexts, actions, rewards):
        self.update_data_pool(contexts, actions, rewards)
        self.set_gram_matrix()


@jax.jit
def sqeuclidean_distance(x, y):
    return jnp.sum((x - y) ** 2)


# Exponential Kernel
@jax.jit
def exp_kernel(gamma, x, y):
    return jnp.exp(- gamma * jnp.sqrt(sqeuclidean_distance(x, y)))


# RBF Kernel
@jax.jit
def rbf_kernel(gamma, x, y):
    return jnp.exp(- gamma * sqeuclidean_distance(x, y))


@jax.jit
def polynomial_kernel(dimension, x, y):
    return (jnp.dot(x, y) + 1) ** dimension


@jax.jit
def linear_kernel(x, y):
    return jnp.dot(x, y) + 1


def gram(func, params, x, y):
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(params, x1, y1))(y))(x)


def gram_linear(func, x, y):
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)


class Kernel:

    def __init__(self):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._param = 1

    def gram_matrix(self, states):
        return self._pairwise(states, states)

    def evaluate(self, state1, state2):
        return self._pairwise(state1, state2)

    def _pairwise(self, X1, X2):
        pass


class Gaussian(Kernel):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(Gaussian, self).__init__(*args)
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._std = self._param

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        return gram(rbf_kernel, 1 / (2 * self._std ** 2), X1, X2)


class Polynomial(Kernel):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(Polynomial, self).__init__(*args)
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._dimension = 2

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        return gram(polynomial_kernel, self._dimension, X1, X2)


class Linear(Kernel):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(Linear, self).__init__(*args)
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        return gram_linear(linear_kernel, X1, X2)

def save_result(dataset_name, random_seed, settings, rollout, online_loss, total_time):
    task_name = 'algo:{}'.format(settings['agent'])
    task_name += '|{}:{}'.format('lambda', settings['lambda'])
    task_name += '|{}:{}'.format('rd', random_seed)
    task_name += '|{}:{}'.format('env', dataset_name)
    task_name += '|{}:{}'.format('rollout', rollout)
    metrics_information = 'online_loss:{}'.format(online_loss)
    metrics_information += '|total_time:{}'.format(total_time)

    result = '{} {}\n'.format(task_name, metrics_information)

    results_dir = 'results/{}/{}_lambda{}'.format(dataset_name, settings['agent'], settings['lambda'])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fname = os.path.join(results_dir, 'metrics.txt')

    with open(fname, 'a') as file:
        file.write(result)


label_cardinality = {
    'scene': 6,
    'yeast': 14,
    'tmc2007': 22,
}


def batch_bandit_experiment(agent, kernel, dataset, seed, settings):
    # Data and env
    X_train, y_train, X_test, y_test, labels = load_dataset(dataset)
    X_all = np.vstack([X_train, X_test])
    y_all = np.vstack([y_train, y_test])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=.33, random_state=seed + 42
    )

    samples = rollout_indices(len(X_train), 'linear', 10)
    # samples = samples[:4]

    # use CRM dataset to initialize agent
    crm_dataset = CRMDataset(seed=seed)
    pi0, _ = make_baselines_skylines(dataset, X_train, y_train, n_jobs=4, skip_skyline=True)
    start = 0
    end = samples[1]
    X = X_train[start:end, :]
    y = y_train[start:end, :]
    sampling_probas = np.array([_[:, 1] for _ in pi0.predict_proba(X)]).T

    crm_dataset.update_from_supervised_dataset(X, y, sampling_probas, n_samples=4)
    k = label_cardinality[dataset]
    env = Environment(crm_dataset, k)
    agent.instantiate(env)

    rollout_losses = []

    t0 = time.time()

    start = samples[1]
    for step, end in tqdm(enumerate(samples[2:])):
        contexts = X_train[start:end, :]
        labels = y_train[start:end, :]
        actions = agent.sample_actions(contexts)
        rewards = env.sample_reward(actions, labels)
        agent.update_agent(contexts, actions, rewards)

        # test
        contexts, labels = X_test, y_test
        actions = agent.sample_actions(contexts)
        rewards = env.sample_reward(actions, labels)
        online_loss = 1 - np.mean(rewards)
        print('Rollout {}, Online Loss: {}'.format(step + 2, online_loss))
        start = end
        t = time.time() - t0
        save_result(dataset, seed, settings, step, online_loss, t)
        rollout_losses.append(online_loss)

    return np.array(rollout_losses)


def bkcub_experiment(dataset, lbd, seed):
    settings = {
        'lambda': lbd,
        'agent': 'BKUCB'
    }

    kernel = Gaussian()
    agent = BatchKernelUCB(settings, kernel)
    return batch_bandit_experiment(agent, kernel, dataset, seed, settings)


def sbpe_experiment(dataset, seed):
    settings = {
        'lambda': 0.,
        'agent': 'BKUCB'
    }

    kernel = Linear()
    agent = SBPE(settings, kernel)
    return batch_bandit_experiment(agent, kernel, dataset, seed, settings)