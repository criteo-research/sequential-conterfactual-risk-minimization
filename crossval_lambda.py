import argparse

import numpy as np
import pandas as pd

from baselines_skylines import make_baselines_skylines
# import jax
# from jax.config import config as jax_config
# jax_config.update("jax_debug_nans", True)

from crm_dataset import CRMDataset
from crm_model import Model
from dataset_utils import load_dataset
from lambda_grid import rollout_indices, run_baseskyline

DEFAULT_LAMBDA_GRID = [1e-3, 1e-1, .5, 0, -.5, -1e-1, -1e-3]


def run_crm(args, X_train, y_train, X_test, y_test, pi0, samples):

    test_sampling_probas = np.array([_[:, 1] for _ in pi0.predict_proba(X_test)]).T
    crm_reward = CRMDataset().update_from_supervised_dataset(
        X_test, y_test, test_sampling_probas, n_samples=4
    ).rewards.sum() / (len(X_test) * y_test.shape[1] * 4)

    crm_losses = np.ones((args.n_reruns, len(samples)))
    crm_rewards = np.ones((args.n_reruns, len(samples)))

    for i in range(args.n_reruns):
        np.random.seed(i * 42 + 1000)
        print("CRM", i, end='')

        crm_model = Model.null_model(X_test.shape[1], y_test.shape[1])
        crm_dataset = CRMDataset()

        start = 0
        for j, end in enumerate(samples):
            print('.', end='')
            # current batch
            X = X_train[start:end, :]
            y = y_train[start:end, :]
            if end > start:
                # CRM play & data collection
                sampling_probas = np.array([_[:, 1] for _ in pi0.predict_proba(X)]).T
                crm_dataset.update_from_supervised_dataset(X, y, sampling_probas, n_samples=args.n_replays)
                # lambda crossval
                best_lambda = crm_model.cross_validate_lambda(
                    crm_dataset, args.n_replays*len(X_train), args.lambda_grid,
                )
                # learning
                crm_model.fit(crm_dataset, lambda_=best_lambda)
            # eval
            crm_losses[i, j] = crm_model.expected_hamming_loss(X_test, y_test)
            crm_rewards[i, j] = crm_reward * len(X)
            # next round
            start = end

        loss = crm_losses.mean(axis=0)[-1]
        reward = crm_rewards.mean(axis=0).sum()
        print("final loss: %.3f" % loss, "cum reward: %d" % reward)
        print('*' * 80)
    return loss, reward


def run_scrm(args, X_train, y_train, X_test, y_test, pi0, samples):

    scrm_losses = np.ones((args.n_reruns, len(samples)))
    scrm_rewards = np.ones((args.n_reruns, len(samples)))

    for i in range(args.n_reruns):
        np.random.seed(i * 42 + 1000)
        print(i, end='')

        scrm_model = Model.null_model(X_test.shape[1], y_test.shape[1])
        scrm_dataset = CRMDataset()

        start = 0
        for j, end in enumerate(samples):
            print("SCRM", '.', end='')
            # current batch
            X = X_train[start:end, :]
            y = y_train[start:end, :]
            if end > start:
                # CRM play & data collection
                if j == 0:
                    sampling_probas = np.array([_[:, 1] for _ in pi0.predict_proba(X)]).T
                else:
                    sampling_probas = scrm_model.predict_proba(X, y)
                scrm_dataset.update_from_supervised_dataset(X, y, sampling_probas, n_samples=args.n_replays)
                # lambda crossval
                best_lambda = scrm_model.cross_validate_lambda(
                    scrm_dataset, args.n_replays * len(X_train), args.lambda_grid,
                )
                # learning
                scrm_model.fit(
                    scrm_dataset,
                    lambda_=best_lambda,
                    verbose=0
                )
            # eval
            scrm_losses[i, j] = scrm_model.expected_hamming_loss(X_test, y_test)
            scrm_rewards[i, j] = CRMDataset().update_from_supervised_dataset(
                X_test, y_test, scrm_model.predict_proba(X_test, np.ones_like(y_test)), n_samples=4
            ).rewards.sum() / (len(X_test) * y_test.shape[1] * 4) * len(X)
            # next round
            start = end
        print()

        loss = scrm_losses.mean(axis=0)[-1]
        reward = scrm_rewards.mean(axis=0).sum()
        print("final loss: %.3f" % loss, "cum reward: %d" % reward)
        print('*' * 80)

    return loss, reward


def export_results(args,
                   baseline_loss, skyline_loss, crm_loss_by_lambda, scrm_loss_by_lambda,
                   baseline_rewards, skyline_rewards, crm_reward_by_lambda, scrm_reward_by_lambda):
    pd.DataFrame(data={
        'lambda': args.lambda_grid,
        'baseline': baseline_loss,
        'CRM': crm_loss_by_lambda,
        'S-CRM': scrm_loss_by_lambda,
        'skyline': skyline_loss,
    }).to_latex('loss_xval_lambda-%s-%d.tex' % (args.rollout_scheme, args.n_rollouts), index=False)
    pd.DataFrame(data={
        'lambda': args.lambda_grid,
        'baseline': baseline_rewards,
        'CRM': crm_reward_by_lambda,
        'S-CRM': scrm_reward_by_lambda,
        'skyline': skyline_rewards,
    }).to_latex('cumreward_xval_lambda-%s-%d.tex' % (args.rollout_scheme, args.n_rollouts), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-rollouts', '-rol', type=int, default=5)
    parser.add_argument('--n-replays', '-rep', type=int, default=4)
    parser.add_argument('--n-reruns', '-rer', type=int, default=5)
    parser.add_argument('--rollout-scheme', '-rs', default='linear', choices=('linear', 'doubling'))
    def parse_lambdas(x): return [float(_) for _ in x.split(',')]
    parser.add_argument('--lambda-grid', '-lg', default=DEFAULT_LAMBDA_GRID, type=parse_lambdas)
    args = parser.parse_args()

    loss_results = {
            'Baseline': [],
            'CRM': [],
            'SCRM': [],
            'Skyline': []
        }
    reward_results = {
            'Baseline': [],
            'CRM': [],
            'SCRM': [],
            'Skyline': []
        }

    datasets = ('scene', 'yeast', 'tmc2007',)
    for dataset in datasets:
        print("DATASET:", dataset)
        args.dataset_name = dataset
        if dataset in ('scene', 'yeast',):
            args.n_reruns = 10
            args.n_rollouts = 5
        else:
            args.n_reruns = 5
            args.n_rollouts = 10
        X_train, y_train, X_test, y_test, labels = load_dataset(args.dataset_name)
        samples = rollout_indices(len(X_train), args.rollout_scheme, args.n_rollouts)
        pi0, pistar = make_baselines_skylines(args.dataset_name, X_train, y_train, n_jobs=4)
        crm_loss, crm_reward = run_crm(args, X_train, y_train, X_test, y_test, pi0, samples)
        scrm_loss, scrm_reward = run_scrm(args, X_train, y_train, X_test, y_test, pi0, samples)
        baseline_rewards, skyline_rewards, baseline_loss, skyline_loss = run_baseskyline(
            args, X_test, y_test, pi0, pistar, samples)
        loss_results['Baseline'] += [baseline_loss[-1]]
        loss_results['CRM'] += [crm_loss]
        loss_results['SCRM'] += [scrm_loss]
        loss_results['Skyline'] += [skyline_loss[-1]]
        reward_results['Baseline'] += [baseline_rewards[-1]]
        reward_results['CRM'] += [crm_reward]
        reward_results['SCRM'] += [scrm_reward]
        reward_results['Skyline'] += [skyline_rewards[-1]]
        print("LOSS:", loss_results)
        print("REWARD:", reward_results)

    pd.DataFrame(data=loss_results, index=datasets).to_latex(
        'loss-xval-lambda-%s-%d.tex' % (args.rollout_scheme, args.n_rollouts), index=True
    )
    pd.DataFrame(data=reward_results, index=datasets).to_latex(
        'rewards-xval-lambda-%s-%d.tex' % (args.rollout_scheme, args.n_rollouts), index=True
    )
