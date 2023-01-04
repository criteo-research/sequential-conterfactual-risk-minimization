from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset import get_dataset_by_name
from utils.utils import LossHistory, online_evaluation, start_experiment, get_logging_data, update_past_data
from src.batch_bandits.batch_bandit import batch_bandit_experiment
from src.crm.scrm import scrm_myopic_experiment
import time

def scrm_experiments(experiment, dataset_name, settings):

    histories = []
    times = []

    for random_seed in range(10):
        t0 = time.time()
        loss_history = experiment(random_seed, dataset_name, settings)
        times.append(time.time() - t0)
        histories.append(loss_history)

    return histories, times


def batch_bandit_experiments(dataset_name, settings):
    batch_bandit_online_losses = []
    batch_bandit_cumulated_losses = []
    times = []

    for random_seed in range(10):
        t0 = time.time()
        online_losses, cumulated_losses = batch_bandit_experiment(random_seed, dataset_name, settings)
        times.append(time.time() - t0)
        batch_bandit_online_losses.append(online_losses)
        batch_bandit_cumulated_losses.append(cumulated_losses)

    return batch_bandit_online_losses, batch_bandit_cumulated_losses, times

dataset_name = 'noisymoons'
settings = {
    'agent': 'Batch-KUCB',
    'kernel': 'polynomial',
    'n_0': 10,
    'M': 5,
    'random_seed': 42,
    'reg_lambda': 1,
}

# Batch K-UCB
batch_bandit_online_losses, batch_bandit_cumulated_losses, batch_bandit_times = batch_bandit_experiments(dataset_name, settings)
batch_bandit_online_losses = np.concatenate(batch_bandit_online_losses, axis=0)
batch_kucb_online_losses, batch_k_ucb_online_losses_std = np.mean(batch_bandit_online_losses, axis=0), np.std(
    batch_bandit_online_losses, axis=0)

batch_bandit_cumulated_losses = np.concatenate(batch_bandit_cumulated_losses, axis=0)
batch_kucb_cumulated_losses, batch_k_ucb_cumulated_losses_std = np.mean(batch_bandit_cumulated_losses, axis=0), np.std(
    batch_bandit_cumulated_losses, axis=0)

# SCRM
settings = {
    'lambda': 0.01,
    'contextual_modelling': 'linear',
    'n_0': 10,
    'M':5,
}

scrm_m_histories, scrm_times = scrm_experiments(scrm_myopic_experiment, dataset_name, settings)

scrm_m_online_losses = np.array([scrm_m_loss_history.online_loss for scrm_m_loss_history in scrm_m_histories])
mean_scrm_m_online_losses = np.mean(scrm_m_online_losses, axis=0)

plt.figure()
plt.title('Loss Evolution over Rollouts')
plt.xlabel('Rollouts')
plt.plot(mean_scrm_m_online_losses, '-.', label='SCRM')
plt.plot(batch_kucb_online_losses, '-', label='Batch-K-UCB')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('test_batch_losses.pdf')

