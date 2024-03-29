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



def batch_bandit_experiments(dataset_name, settings, lambda_grid):
    batch_bandit_online_losses = []
    times = []

    for random_seed in range(10):
        t0 = time.time()

        best_loss = np.inf
        best_onlines_losses = None

        # a posteriori selection

        for lbd in lambda_grid:

            settings['lambda'] = lbd
            online_losses = batch_bandit_experiment(random_seed, dataset_name, settings)
            times.append(time.time() - t0)

            loss = np.squeeze(online_losses)[-1]

            if loss < best_loss:
                best_loss = loss
                best_onlines_losses = online_losses

            batch_bandit_online_losses.append(best_onlines_losses)

    return batch_bandit_online_losses, times

settings = {
    'contextual_modelling': 'linear',
    'kernel': 'gaussian',
    'n_0': 100,
    'M': 10,
    'data': 'geometrical',
    'validation': False,
    'lambda':0.
}

bkucb_regularization_lambda_grid = np.array([1e2, 1e1, 1e0])


def scrm_vs_baselines(results, dataset_name, settings):



    # Batch K-UCB
    settings['agent'] = 'BKUCB'
    bkucb_online_losses, bkucb_times = batch_bandit_experiments(
        dataset_name, settings, bkucb_regularization_lambda_grid)
    bkucb_online_losses = np.concatenate(bkucb_online_losses, axis=0)
    bkucb_online_losses, bkucb_online_losses_std = np.mean(bkucb_online_losses, axis=0), np.std(
        bkucb_online_losses, axis=0)



    # Report performances
    bkucb_perf, bkucb_std = bkucb_online_losses[-1], bkucb_online_losses_std[-1]



    results['dataset'] += [dataset_name]

    results['BKUCB'] += ['$%.3f \pm %.3f$' % (bkucb_perf, bkucb_std)]


    return results

results = defaultdict(list)

for dataset_name in ['pricing', 'advertising']:
    results = scrm_vs_baselines(results, dataset_name, settings)

df = pd.DataFrame(data=results)
df.to_latex(
    'compare_baselines_scrm_bkucb_1_continuous.tex', index=False, column_format='r', escape=False
)

print('-' * 80)
print(df)
print('-' * 80)



