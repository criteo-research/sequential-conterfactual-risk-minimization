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


settings = {
    'contextual_modelling': 'linear',
    'kernel': 'polynomial',
    'n_0': 10,
    'M': 10,
    'data': 'geometrical',
    'validation': False,
    'lambda':0.
}

scrm_variance_penalty_lambda_grid = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])


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


    # RL Baselines


    # Report performances
    scrm_perf, scrm_std = mean_scrm_m_online_losses[-1], std_scrm_m_losses[-1]
    crm_perf, crm_std = mean_crm_online_losses[-1], std_crm_online_loss[-1]
    # ppo_perf, ppo_std =
    # trpo_perf, trpo_std =


    results['dataset'] += [dataset_name]
    results['SCRM'] += ['$%.3f \pm %.3f$' % (scrm_perf, scrm_std)]
    results['CRM'] += ['$%.3f \pm %.3f$' % (crm_perf, crm_std)]

    # results['PPO'] += ['$%.3f \pm %.3f$' % (ppo_perf, ppo_std)]
    # results['TRPO'] += ['$%.3f \pm %.3f$' % (trpo_perf, trpo_std)]

    return results

results = defaultdict(list)

for dataset_name in ['pricing', 'advertising']:
    results = scrm_vs_baselines(results, dataset_name, settings)

df = pd.DataFrame(data=results)
df.to_latex(
    'compare_baselines_scrm_counterfactual_continuous.tex', index=False, column_format='r', escape=False
)

print('-' * 80)
print(df)
print('-' * 80)



