import numpy as np
import os
import sys


class LossHistory(object):

    def __init__(self, name):
        self.name = name
        self.crm_loss = []
        self.online_loss = []
        self.betas = []
        self.n_samples = []
        self.n_actions = []
        self.cumulated_loss = []
        self.regret = []

    def update(self, beta, online_loss, regret, crm_loss, cumulated_losses, n_samples):
        self.betas += [beta]
        self.online_loss += [online_loss]
        self.crm_loss += [crm_loss]
        self.n_samples += [n_samples]
        self.cumulated_loss += [np.sum(self.cumulated_loss) + cumulated_losses]
        self.regret += [np.sum(self.regret) + regret * n_samples]

    def show_last(self):
        print(
            '<', self.name,
            'CRM loss: %.5f' % self.crm_loss[-1],
            'Online loss: %.5f' % self.online_loss[-1],
            '|theta|=%.2f' % np.sqrt((self.betas[-1] ** 2).sum()),
            'n=%d' % sum(self.n_samples),
            '>',
            file=sys.stderr
        )

def get_logging_data(n_samples, dataset):

    actions, contexts, losses, propensities, potentials = dataset.sample_logged_data(n_samples)
    logging_data = actions, contexts, losses, propensities

    return logging_data

def update_past_data(data, samples):
    return np.hstack([data, samples])

def online_evaluation(optimized_param, contextual_modelling, dataset, random_seed):
    rng = np.random.RandomState(random_seed)
    contexts, potentials = dataset.test_data
    contextual_param = contextual_modelling.get_parameter(optimized_param, contexts)
    size = contexts.shape[0]
    losses = []

    for i in range(10):
        sampled_actions = rng.normal(contextual_param, dataset.logging_scale, size)
        losses += [dataset.get_losses_from_actions(potentials, sampled_actions)]

    losses_array = np.stack(losses, axis=0)
    var_pi = np.mean(np.var(losses_array, axis=0))
    var_context = np.var(np.mean(losses_array, axis=1))
    return np.mean(losses_array), np.std(np.mean(losses_array, axis=0))


def start_experiment(random_seed, dataset, name):
    print(
        '***', 'EXPERIMENT', name,
        'Random seed: %i' % random_seed,
        'Dataset: %s' % dataset.name,
        '***',
        file=sys.stderr
    )