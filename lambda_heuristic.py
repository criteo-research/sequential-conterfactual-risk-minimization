import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

# from jax.config import config as jax_config
# jax_config.update("jax_debug_nans", True)

from joblib import Parallel, delayed

from dataset_utils import load_dataset
from baselines_skylines import make_baselines_skylines, stochastic_hamming_loss
from crm_dataset import CRMDataset
from crm_model import Model

DEFAULT_LAMBDA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]


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
    parser.add_argument('--n-rollouts', '-rol', type=int, default=10)
    parser.add_argument('--n-replays', '-rep', type=int, default=6)
    parser.add_argument('--n-reruns', '-rer', type=int, default=10)
    parser.add_argument('--rollout-scheme', '-rs', default='linear', choices=('linear', 'doubling'))
    def parse_lambdas(x): return [float(_) for _ in x.split(',')]
    parser.add_argument('--lambda-grid', '-lg', default=DEFAULT_LAMBDA_GRID, type=parse_lambdas)
    parser.add_argument('--to', '-to', default='')
    parser.add_argument('--n-jobs', '-j', default=1, type=int)
    parser.add_argument('--crm', default=False, action='store_true')
    parser.add_argument('--truevar', default=False, action='store_true')
    parser.add_argument('--ips-ix', default=False, action='store_true')

    args = parser.parse_args()

    results = defaultdict(list)

    for dataset in args.datasets:
        print("DATASET:", dataset)

        X_train, y_train, X_test, y_test, labels = load_dataset(dataset)
        pi0, pistar = make_baselines_skylines(dataset, X_train, y_train, n_jobs=4)

        samples = rollout_indices(len(X_train), args.rollout_scheme, args.n_rollouts)
        print("rollout @", samples)

        def _run(l):
            return run_crm(X_train, y_train, X_test, y_test, pi0, samples,
                           n_reruns=args.n_reruns, n_replays=args.n_replays, lambda_grid=args.lambda_grid,
                           autotune_lambda=l is None, lambda_=l, scrm=not args.crm,
                           truevar=args.truevar, ips_ix=args.ips_ix)

        scrm_loss_autotuned_lambda_mean, scrm_loss_autotuned_lambda_std = _run(None)
        print('SCRM loss w. autotune lamda: %.5f +/- %.5f' % (
            scrm_loss_autotuned_lambda_mean, scrm_loss_autotuned_lambda_std)
        )

        all_losses_stds = Parallel(n_jobs=args.n_jobs)(delayed(_run)(l) for l in args.lambda_grid)
        print('MAP:', sorted(zip(all_losses_stds, args.lambda_grid)))
        scrm_best_loss_a_posteriori_mean, scrm_best_loss_a_posteriori_std = sorted(all_losses_stds)[0]
        print('SCRM best loss a posteriori: %.5f +/- %.5f' % (
            scrm_best_loss_a_posteriori_mean, scrm_best_loss_a_posteriori_std)
        )

        results['dataset'] += [dataset]
        results['Baseline'] += ['$%.5f$' % stochastic_hamming_loss(pi0, X_test, y_test)]
        results['SCRM (MAP)'] += ['$%.5f \pm %.5f$' % (scrm_best_loss_a_posteriori_mean, scrm_best_loss_a_posteriori_std)]
        results['SCRM (auto-tune)'] += ['$%.5f \pm %.5f$' % (scrm_loss_autotuned_lambda_mean, scrm_loss_autotuned_lambda_std)]
        results['Skyline'] += ['$%.5f$' % stochastic_hamming_loss(pistar, X_test, y_test)]

    df = pd.DataFrame(data=results)
    df.to_latex(
        'lambda_heuristic_discrete-%s-rs_%s-ro_%d-rr_%d.tex' % (
            args.to, args.rollout_scheme, args.n_rollouts, args.n_reruns
        ), index=False, column_format='r', escape=False
    )
    print('-'*80)
    print(df)
    print('-'*80)
