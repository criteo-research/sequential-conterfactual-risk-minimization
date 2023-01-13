from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.dataset import get_dataset_by_name
from utils.utils import LossHistory, online_evaluation, start_experiment, get_logging_data, update_past_data
from src.batch_bandits.batch_bandit import batch_bandit_experiment
from src.crm.scrm import scrm_myopic_experiment
from src.crm.crm import repeated_crm_experiment

import time

from collections import defaultdict
import pandas as pd

def counterfactual_experiments(experiment, dataset_name, settings, lambda_grid):

    histories = []
    times = []

    for random_seed in range(10):

        best_loss = np.inf
        best_history = {}

        # a posteriori selection

        for lbd in lambda_grid:

            settings['lambda'] = lbd
            t0 = time.time()
            loss_history = experiment(random_seed, dataset_name, settings)
            times.append(time.time() - t0)
            loss = loss_history.online_loss[-1]

            if loss < best_loss:
                best_loss = loss
                best_history = loss_history

        histories.append(best_history)

    return histories, times


def batch_bandit_experiments(dataset_name, settings, lambda_grid):
    batch_bandit_online_losses = []
    batch_bandit_cumulated_losses = []
    times = []

    for random_seed in range(10):
        t0 = time.time()

        best_loss = np.inf
        best_onlines_losses = None
        best_cumulated_losses = None

        # a posteriori selection

        for lbd in lambda_grid:

            settings['lambda'] = lbd
            online_losses, cumulated_losses = batch_bandit_experiment(random_seed, dataset_name, settings)
            times.append(time.time() - t0)

            loss = np.squeeze(online_losses)[-1]

            if loss < best_loss:
                best_loss = loss
                best_onlines_losses = online_losses
                best_cumulated_losses = cumulated_losses

            batch_bandit_online_losses.append(best_onlines_losses)
            batch_bandit_cumulated_losses.append(best_cumulated_losses)

    return batch_bandit_online_losses, batch_bandit_cumulated_losses, times

settings = {
    'contextual_modelling': 'linear',
    'kernel': 'polynomial',
    'n_0': 100,
    'M': 10,
    'data': 'geometrical',
    'validation': False,
    'lambda':0.
}

scrm_variance_penalty_lambda_grid = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
bkucb_regularization_lambda_grid = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1])


def scrm_vs_baselines(results, dataset_name, settings):

    # SCRM
    scrm_m_histories, scrm_times = counterfactual_experiments(scrm_myopic_experiment, dataset_name, settings, scrm_variance_penalty_lambda_grid)

    scrm_m_online_losses = np.array([scrm_m_loss_history.online_loss for scrm_m_loss_history in scrm_m_histories])
    mean_scrm_m_online_losses = np.mean(scrm_m_online_losses, axis=0)
    std_scrm_m_losses = np.std(scrm_m_online_losses, axis=0)

    # CRM
    crm_histories, crm_times = counterfactual_experiments(repeated_crm_experiment, dataset_name, settings, scrm_variance_penalty_lambda_grid)
    crm_online_losses = np.array([crm_loss_history.online_loss for crm_loss_history in crm_histories])
    mean_crm_online_losses = np.mean(crm_online_losses, axis=0)
    std_crm_online_loss = np.std(crm_online_losses, axis=0)

    # Batch K-UCB
    settings['agent'] = 'BKUCB'
    bkucb_online_losses, bkucb_cumulated_losses, bkucb_times = batch_bandit_experiments(
        dataset_name, settings, bkucb_regularization_lambda_grid)
    bkucb_online_losses = np.concatenate(bkucb_online_losses, axis=0)
    bkucb_online_losses, bkucb_online_losses_std = np.mean(bkucb_online_losses, axis=0), np.std(
        bkucb_online_losses, axis=0)

    # SBPE
    settings['agent'] = 'SBPE'
    sbpe_online_losses, sbpe_cumulated_losses, sbpe_times = batch_bandit_experiments(
        dataset_name, settings, [0.])
    sbpe_online_losses = np.concatenate(sbpe_online_losses, axis=0)
    sbpe_online_losses, sbpe_online_losses_std = np.mean(sbpe_online_losses, axis=0), np.std(
        sbpe_online_losses, axis=0)

    # RL Baselines


    # Report performances
    scrm_perf, scrm_std = mean_scrm_m_online_losses[-1], std_scrm_m_losses[-1]
    crm_perf, crm_std = mean_crm_online_losses[-1], std_crm_online_loss[-1]
    bkucb_perf, bkucb_std = bkucb_online_losses[-1], bkucb_online_losses_std[-1]
    sbpe_perf, sbpe_std = sbpe_online_losses[-1], sbpe_online_losses_std[-1]
    # ppo_perf, ppo_std =
    # trpo_perf, trpo_std =


    results['dataset'] += [dataset_name]
    results['SCRM'] += ['$%.3f \pm %.3f$' % (scrm_perf, scrm_std)]
    results['CRM'] += ['$%.3f \pm %.3f$' % (crm_perf, crm_std)]
    results['BKUCB'] += ['$%.3f \pm %.3f$' % (bkucb_perf, bkucb_std)]
    results['SPBE'] += ['$%.3f \pm %.3f$' % (sbpe_perf, sbpe_std)]

    # results['PPO'] += ['$%.3f \pm %.3f$' % (ppo_perf, ppo_std)]
    # results['TRPO'] += ['$%.3f \pm %.3f$' % (trpo_perf, trpo_std)]

    return results

results = defaultdict(list)

for dataset_name in ['pricing', 'advertising']:
    results = scrm_vs_baselines(results, dataset_name, settings)

df = pd.DataFrame(data=results)
df.to_latex(
    'compare_baselines_scrm_continuous.tex', index=False, column_format='r', escape=False
)

print('-' * 80)
print(df)
print('-' * 80)



