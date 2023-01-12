import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

# from jax.config import config as jax_config
# jax_config.update("jax_debug_nans", True)

from joblib import Parallel, delayed

from compare_crm_scrm import DEFAULT_LAMBDA_GRID, rollout_indices, add_common_arguments, run_crm_multiple_times, \
    save_config
from dataset_utils import load_dataset
from baselines_skylines import make_baselines_skylines, stochastic_hamming_loss
from crm_dataset import CRMDataset
from crm_model import Model


def run_crm(X_train, y_train, X_test, y_test, pi0, samples, n_reruns, n_replays, lambda_grid,
            scrm: bool = False,
            autotune_lambda: bool = False, lambda_: float = .01, ips_ix: bool = False, truevar: bool = False):
    # lambda_ is None => autotune

    crm_losses = []
    for i in range(n_reruns):
        np.random.seed(i * 42 + 1000)
        print('.', end='')

        crm_model = Model.null_model(X_test.shape[1], y_test.shape[1])
        crm_dataset = CRMDataset()

        start = 0
        for j, end in enumerate(samples):
            # current batch
            X = X_train[start:end, :]
            y = y_train[start:end, :]
            if end > start:
                # CRM play & data collection
                if not scrm or (j == 0):
                    sampling_probas = np.array([_[:, 1] for _ in pi0.predict_proba(X)]).T
                else:
                    sampling_probas = crm_model.predict_proba(X, y)
                crm_dataset.update_from_supervised_dataset(X, y, sampling_probas, n_samples=n_replays)
                # xval lambda if needed
                if autotune_lambda:
                    lambda_ = Model.autotune_lambda(
                        crm_dataset, crm_model.d, crm_model.k, grid=lambda_grid,
                        sequential_dependence=truevar, ips_ix=ips_ix, snips=not ips_ix
                    )
                # learning
                if scrm or j == len(samples) - 1:
                    crm_model.fit(crm_dataset, lambda_=lambda_, sequential_dependence=truevar, ips_ix=ips_ix, snips=not ips_ix)
            # next round
            start = end

        # final eval
        crm_losses += [crm_model.expected_hamming_loss(X_test, y_test)]
    print()
    return np.mean(crm_losses), np.std(crm_losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def parse_datasets(x): return x.split(',')
    parser.add_argument('datasets', type=parse_datasets)
    add_common_arguments(parser)
    def parse_lambdas(x): return [float(_) for _ in x.split(',')]
    parser.add_argument('--lambda-grid', '-lg', default=DEFAULT_LAMBDA_GRID, type=parse_lambdas)
    args = parser.parse_args()

    results = defaultdict(list)

    for dataset in args.datasets:
        print("DATASET:", dataset)

        X_train, y_train, X_test, y_test, labels = load_dataset(dataset)
        X_all = np.vstack([X_train, X_test])
        y_all = np.vstack([y_train, y_test])

        autotuned_scrm_losses = run_crm_multiple_times(X_all, y_all, dataset, args,
                                                       scrm=True, lamda=None, autotune_lambda=True)
        autotuned_scrm_losses_mean = np.mean(autotuned_scrm_losses)
        autotuned_scrm_losses_std = np.std(autotuned_scrm_losses)
        print('SCRM loss w. autotune lamda: %.3f +/- %.3f' % (
            autotuned_scrm_losses_mean, autotuned_scrm_losses_std)
        )

        results['dataset'] += [dataset]
        results['SCRM (auto-tune)'] += ['$%.3f \pm %.3f$' % (autotuned_scrm_losses_mean, autotuned_scrm_losses_std)]

    df = pd.DataFrame(data=results)
    fn = 'lambda_heuristic_discrete-%s-all-datasets.tex' % args.prefix
    df.to_latex(fn, index=False, column_format='r', escape=False)
    save_config(args, "alldatasets", 'lambda_heuristic_discrete')

