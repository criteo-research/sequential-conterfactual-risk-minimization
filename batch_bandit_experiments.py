import numpy as np

from collections import defaultdict
import pandas as pd
from discrete_batch_bandits import bkcub_experiment, sbpe_experiment

lambdas = [1e0, 1e1, 1e2]

def bkucb_experiments(dataset_name, lambda_grid):

    loss_std = 0
    best_loss = np.inf
    print('-' * 80)
    print('BKCUB experiment, dataset {}'.format(dataset_name))

    for lbd in lambda_grid:
        print('-' * 80)
        print('Lambda {}'.format(lbd))

        seeds_losses = []


        for random_seed in range(10):
            # a posteriori selection
            print('-' * 80)
            print('Seed {}'.format(random_seed))
            rollout_losses = bkcub_experiment(dataset_name, lbd, random_seed)

            final_loss = np.squeeze(rollout_losses)[-1]
            seeds_losses.append(final_loss)

        seed_loss = np.mean(seeds_losses)

        if seed_loss < best_loss:
            best_loss = seed_loss
            loss_std = np.std(seeds_losses)

    return best_loss, loss_std


def sbpe_experiments(dataset_name):
    seeds_losses = []

    print('-' * 80)
    print('SBPE experiment, dataset {}'.format(dataset_name))

    for random_seed in range(10):
        print('-' * 80)
        print('Seed {}'.format(random_seed))
        # a posteriori selection
        rollout_losses = sbpe_experiment(dataset_name, random_seed)

        final_loss = np.squeeze(rollout_losses)[-1]
        seeds_losses.append(final_loss)

    loss = np.mean(seeds_losses)
    loss_std = np.std(seeds_losses)

    return loss, loss_std

if __name__ == '__main__':
    results = defaultdict(list)

    dataset_name = 'scene'
    # Report performances
    bkucb_perf, bkucb_std = bkucb_experiments(dataset_name, lambdas)
    sbpe_perf, sbpe_std = sbpe_experiments(dataset_name)


    results['dataset'] += [dataset_name]

    results['BKUCB'] += ['$%.3f \pm %.3f$' % (bkucb_perf, bkucb_std)]
    results['SBPE'] += ['$%.3f \pm %.3f$' % (sbpe_perf, sbpe_std)]

    df = pd.DataFrame(data=results)
    df.to_latex(
        'batch_bandit_results.tex', index=False, column_format='r', escape=False
    )

    print('-' * 80)
    print(df)
    print('-' * 80)

    dataset_name = 'yeast'
    # Report performances
    bkucb_perf, bkucb_std = bkucb_experiments(dataset_name, lambdas)
    sbpe_perf, sbpe_std = sbpe_experiments(dataset_name)

    results['dataset'] += [dataset_name]

    results['BKUCB'] += ['$%.3f \pm %.3f$' % (bkucb_perf, bkucb_std)]
    results['SBPE'] += ['$%.3f \pm %.3f$' % (sbpe_perf, sbpe_std)]

    df = pd.DataFrame(data=results)
    df.to_latex(
        'batch_bandit_results.tex', index=False, column_format='r', escape=False
    )

    print('-' * 80)
    print(df)
    print('-' * 80)



