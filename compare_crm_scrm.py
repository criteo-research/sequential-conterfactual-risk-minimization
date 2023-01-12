import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

# from jax.config import config as jax_config
# jax_config.update("jax_debug_nans", True)

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from dataset_utils import load_dataset
from baselines_skylines import make_baselines_skylines, stochastic_hamming_loss
from crm_dataset import CRMDataset
from crm_model import Model, EpsilonGreedyModel

DEFAULT_LAMBDA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]


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


def compute_baseskylines_expected_performance(X_all, y_all, args, dataset):
    pi0_losses = []
    pistar_losses = []
    for seed in range(args.n_reruns):
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=args.test_size, random_state=seed + 42
        )
        pi0, pistar = make_baselines_skylines(dataset, X_train, y_train, n_jobs=4)
        pi0_losses += [stochastic_hamming_loss(pi0, X_test, y_test, epsilon=args.epsilon)]
        pistar_losses += [stochastic_hamming_loss(pistar, X_test, y_test, epsilon=args.epsilon)]
    return np.mean(pi0_losses), np.std(pi0_losses), np.mean(pistar_losses), np.std(pistar_losses)


def run_crm_once(X_all, y_all, dataset_name, args, seed,
                 lambda_: float, scrm: bool, autotune_lambda: bool = False,
                 trace_loss_evolution: bool = False):

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=seed + 42
    )
    pi0, _ = make_baselines_skylines(dataset_name, X_train, y_train, n_jobs=4, skip_skyline=True)
    samples = rollout_indices(len(X_train), args.rollout_scheme, args.n_rollouts)

    crm_model = EpsilonGreedyModel.null_model(X_test.shape[1], y_test.shape[1], epsilon=args.epsilon)
    crm_dataset = CRMDataset(seed=seed)

    loss_evolution = []
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
            crm_dataset.update_from_supervised_dataset(X, y, sampling_probas, n_samples=args.n_replays)
            # xval lambda if needed
            if autotune_lambda:
                lambda_ = Model.autotune_lambda(
                    crm_dataset, crm_model.d, crm_model.k, grid=args.lambda_grid,
                    sequential_dependence=args.truevar, ips_ix=args.ips_ix and scrm, snips=not args.ips_ix
                )
            # learning
            if scrm or j == len(samples) - 1 or trace_loss_evolution:
                crm_model.fit(crm_dataset, lambda_=lambda_,
                              sequential_dependence=args.truevar, ips_ix=args.ips_ix, snips=not args.ips_ix)
        # per rollout eval
        if trace_loss_evolution:
            loss_evolution += [crm_model.expected_hamming_loss(X_test, y_test)]
        # next round
        start = end

    # final eval
    if trace_loss_evolution:
        return loss_evolution
    return crm_model.expected_hamming_loss(X_test, y_test)


def run_crm_multiple_times(X_all, y_all, dataset_name, args,
                           lamda, scrm: bool, autotune_lambda: bool = False,
                           trace_loss_evolution: bool = False):
    return Parallel(n_jobs=args.n_jobs)(delayed(run_crm_once)(
        X_all, y_all, dataset_name, args, seed, lamda, scrm,
        autotune_lambda=autotune_lambda, trace_loss_evolution=trace_loss_evolution
    ) for seed in range(args.n_reruns))


def optimize_lambda_a_posteriori(X_all, args, dataset, y_all, scrm):
    best_crm_losses = None
    best_crm_mean_loss = 1.
    best_crm_lambda = None
    for lamda in args.lambda_grid:
        crm_losses = run_crm_multiple_times(X_all, y_all, dataset, args, lamda, scrm=scrm)
        if np.mean(crm_losses) < best_crm_mean_loss:
            best_crm_losses = crm_losses
            best_crm_lambda = lamda
            best_crm_mean_loss = np.mean(crm_losses)
            #print('best:', np.mean(best_crm_losses), np.std(best_crm_losses), best_crm_lambda)
    return np.mean(best_crm_losses), np.std(best_crm_losses), best_crm_lambda


def save_config(args, dataset, basename, supinfo: dict = {}):
    fn = '%s-%s-%s.cfg' % (
            basename, args.prefix, dataset
    )
    with open(fn, 'w') as fd:
        for k, v in args.__dict__.items():
            fd.write('%s:%s\n' % (k, v))
        for k, v in supinfo.items():
            fd.write('%s:%s\n' % (k, v))
    print('wrote', fn)


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--test-size', '-ts', type=float, default=.33)
    parser.add_argument('--n-rollouts', '-rol', type=int, default=10)
    parser.add_argument('--n-replays', '-rep', type=int, default=6)
    parser.add_argument('--n-reruns', '-rer', type=int, default=10)
    parser.add_argument('--rollout-scheme', '-rs', default='linear', choices=('linear', 'doubling'))
    parser.add_argument('--epsilon', '-eps', default=.1, type=float)
    parser.add_argument('--n-jobs', '-j', default=6, type=int)
    parser.add_argument('--truevar', default=False, action='store_true')
    parser.add_argument('--ips-ix', default=False, action='store_true')
    parser.add_argument('--prefix', '-p', default='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    def parse_datasets(x): return x.split(',')
    parser.add_argument('datasets', type=parse_datasets)
    def parse_lambdas(x): return [float(_) for _ in x.split(',')]
    parser.add_argument('--lambda-grid', '-lg', default=DEFAULT_LAMBDA_GRID, type=parse_lambdas)
    args = parser.parse_args()

    results = defaultdict(list)

    for dataset in args.datasets:
        print("DATASET:", dataset)

        X_train, y_train, X_test, y_test, labels = load_dataset(dataset)
        X_all = np.vstack([X_train, X_test])
        y_all = np.vstack([y_train, y_test])

        mean_crm_loss, std_crm_loss, best_crm_lambda = optimize_lambda_a_posteriori(X_all, args, dataset, y_all, False)
        print(' CRM best loss a posteriori: %.3f +/- %.3f (%f)' % (
            mean_crm_loss, std_crm_loss, best_crm_lambda
        ))
        mean_scrm_loss, std_scrm_loss, best_scrm_lambda = optimize_lambda_a_posteriori(X_all, args, dataset, y_all, True)
        print('SCRM best loss a posteriori: %.3f +/- %.3f (%f)' % (
            mean_scrm_loss, std_scrm_loss, best_scrm_lambda
        ))

        mean_pi0_loss, std_pi0_loss, mean_pistar_loss, std_pistar_loss = \
            compute_baseskylines_expected_performance(X_all, y_all, args, dataset)
        print('Baseline loss              : %.3f +/- %.3f' % (
            mean_pi0_loss, std_pi0_loss
        ))
        print('Skyline loss               : %.3f +/- %.3f' % (
            mean_pistar_loss, std_pistar_loss
        ))

        results['dataset']  += [dataset]
        results['Baseline'] += ['$%.3f \pm %.3f$' % (mean_pi0_loss, std_pi0_loss)]
        results['CRM']      += ['$%.3f \pm %.3f$' % (mean_crm_loss, std_crm_loss)]
        results['SCRM']     += ['$%.3f \pm %.3f$' % (mean_scrm_loss, std_scrm_loss)]
        results['Skyline']  += ['$%.3f \pm %.3f$' % (mean_pistar_loss, std_pistar_loss)]

        df = pd.DataFrame(data=results)
        df.to_latex(
            'compare_crm_scrm_discrete-%s-%s.tex' % (
                args.prefix, dataset
            ), index=False, column_format='r', escape=False
        )
        save_config(args, dataset, 'compare_crm_scrm_discrete',
                    {'best_lambda_crm': best_crm_lambda, 'best_lambda_scrm': best_scrm_lambda})
