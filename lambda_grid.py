import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import jax
# from jax.config import config as jax_config
# jax_config.update("jax_debug_nans", True)

from dataset_utils import load_dataset
from baselines_skylines import make_baselines_skylines, stochastic_hamming_loss
from crm_dataset import CRMDataset
from crm_model import Model

DEFAULT_LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, .5, 1,
                       0,
                       -1e-4, -1e-3, -1e-2, -1e-1, -.5, -1]


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


def run_crm(args, X_train, y_train, X_test, y_test, pi0, samples):

    test_sampling_probas = np.array([_[:, 1] for _ in pi0.predict_proba(X_test)]).T
    crm_reward = CRMDataset().update_from_supervised_dataset(
        X_test, y_test, test_sampling_probas, n_samples=4
    ).rewards.sum() / (len(X_test) * y_test.shape[1] * 4)

    crm_loss_by_lambda = []
    crm_reward_by_lambda = []

    for lambda_ in args.lambda_grid:
        print('CRM', 'lambda:', lambda_, '*' * 80)

        crm_losses = np.ones((args.n_reruns, len(samples)))
        crm_rewards = np.ones((args.n_reruns, len(samples)))

        for i in range(args.n_reruns):
            np.random.seed(i * 42 + 1000)
            print(i, end='')

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
                    # learning
                    crm_model.fit(crm_dataset, lambda_=lambda_)
                # eval
                crm_losses[i, j] = crm_model.expected_hamming_loss(X_test, y_test)
                crm_rewards[i, j] = crm_reward * len(X)
                # next round
                start = end
            print()

        crm_loss_by_lambda += [crm_losses.mean(axis=0)[-1]]
        crm_reward_by_lambda += [crm_rewards.mean(axis=0).sum()]
        print("final loss: %.3f" % crm_loss_by_lambda[-1], "cum reward: %d" % crm_reward_by_lambda[-1])
        print('*' * 80)
    return crm_loss_by_lambda, crm_reward_by_lambda


def run_scrm(args, X_train, y_train, X_test, y_test, pi0, samples):
    ## S-CRM
    scrm_loss_by_lambda = []
    scrm_reward_by_lambda = []
    for lambda_ in args.lambda_grid:
        print('SCRM', 'lambda:', lambda_, '*' * 80)

        scrm_losses = np.ones((args.n_reruns, len(samples)))
        scrm_rewards = np.ones((args.n_reruns, len(samples)))

        for i in range(args.n_reruns):
            np.random.seed(i * 42 + 1000)
            print(i, end='')

            scrm_model = Model.null_model(X_test.shape[1], y_test.shape[1])
            scrm_dataset = CRMDataset()

            start = 0
            for j, end in enumerate(samples):
                print('.', end='')
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
                    # learning
                    scrm_model.fit(
                        scrm_dataset,
                        lambda_=lambda_,
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

        scrm_loss_by_lambda += [scrm_losses.mean(axis=0)[-1]]
        scrm_reward_by_lambda += [scrm_rewards.mean(axis=0).sum()]
        print("final loss: %.3f" % crm_loss_by_lambda[-1], "cum reward: %d" % crm_reward_by_lambda[-1])
        print('*' * 80)

    return scrm_loss_by_lambda, scrm_reward_by_lambda


def run_baseskyline(args, X_test, y_test, pi0, pistar, samples):
    baseline_reward = np.mean([
        CRMDataset().update_from_supervised_dataset(
            X_test, y_test,
            np.array([_[:, 1] for _ in pi0.predict_proba(X_test)]).T, n_samples=4
        ).rewards.sum() / (len(X_test) * y_test.shape[1] * 4)
        for _ in range(args.n_reruns)])
    baseline_rewards = np.ones_like(args.lambda_grid) * baseline_reward * samples[-1]
    skyline_reward = np.mean([
        CRMDataset().update_from_supervised_dataset(
            X_test, y_test,
            np.array([_[:, 1] for _ in pistar.predict_proba(X_test)]).T, n_samples=4
        ).rewards.sum() / (len(X_test) * y_test.shape[1] * 4)
        for _ in range(args.n_reruns)])
    skyline_rewards = np.ones_like(args.lambda_grid) * skyline_reward * samples[-1]
    # map_skyline_reward = np.mean([
    #     CRMDataset().update_from_supervised_dataset(
    #         X_test, y_test,
    #         pistar.predict(X_test), n_samples=1
    #     ).rewards.sum() / (len(X_test) * y_test.shape[1] * 1)
    #     for _ in range(args.n_reruns)])
    # map_skyline_rewards = map_skyline_reward * samples[-1]
    baseline_loss = np.ones_like(args.lambda_grid) * stochastic_hamming_loss(pi0, X_test, y_test)
    skyline_loss = np.ones_like(args.lambda_grid) * stochastic_hamming_loss(pistar, X_test, y_test)

    return baseline_rewards, skyline_rewards, baseline_loss, skyline_loss


def export_results(args,
                   baseline_loss, skyline_loss, crm_loss_by_lambda, scrm_loss_by_lambda,
                   baseline_rewards, skyline_rewards, crm_reward_by_lambda, scrm_reward_by_lambda):
    pd.DataFrame(data={
        'lambda': args.lambda_grid,
        'baseline': baseline_loss,
        'CRM': crm_loss_by_lambda,
        'S-CRM': scrm_loss_by_lambda,
        'skyline': skyline_loss,
    }).to_latex('loss_by_lambda-%s-%s-%d.tex' % (args.rollout_scheme, args.dataset_name, args.n_rollouts), index=False)
    pd.DataFrame(data={
        'lambda': args.lambda_grid,
        'baseline': baseline_rewards,
        'CRM': crm_reward_by_lambda,
        'S-CRM': scrm_reward_by_lambda,
        'skyline': skyline_rewards,
    }).to_latex('cumreward_by_lambda-%s-%s-%d.tex' % (args.rollout_scheme, args.dataset_name, args.n_rollouts), index=False)
    ##
    fig, (ax1, ax2) = plt.subplots(ncols=2, constrained_layout=True, figsize=(16, 4))
    ax1.set_title('Final Loss by $\lambda$ - %s dataset' % args.dataset_name)
    ax1.set_xlabel('$\lambda$')
    ax1.set_xscale('symlog')
    ax1.plot(args.lambda_grid, baseline_loss, '--', label='Baseline / $\pi_0$')
    ax1.plot(args.lambda_grid, crm_loss_by_lambda, 'o', label='CRM / $\pi_0$')
    ax1.plot(args.lambda_grid, scrm_loss_by_lambda, 'o', label='S-CRM / $\pi_t$')
    ax1.plot(args.lambda_grid, skyline_loss, '--', label='Skyline / $\pi^*$')
    ax1.set_ylabel('Hamming Loss')
    ax1.legend(loc=(1.025, .5))
    ax2.set_title('Cumulated Reward by $\lambda$ - %s dataset' % args.dataset_name)
    ax2.set_xlabel('$\lambda$')
    ax2.plot(args.lambda_grid, baseline_rewards, '--', label='Baseline / $\pi_0$')
    ax2.plot(args.lambda_grid, crm_reward_by_lambda, 'o', label='CRM / $\pi_0$')
    ax2.plot(args.lambda_grid, scrm_reward_by_lambda, 'o', label='S-CRM / $\pi_t$')
    ax2.plot(args.lambda_grid, skyline_rewards, '--', label='Skyline / $\pi^*$')
    ax2.set_xscale('symlog')
    ax2.set_ylabel('Cumulated Reward')
    ax2.legend(loc=(1.025, .5))
    plt.savefig('lambda_grid-%s-%s-%d.png' % (args.rollout_scheme, args.dataset_name, args.n_rollouts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name')
    parser.add_argument('--n-rollouts', '-rol', type=int, default=10)
    parser.add_argument('--n-replays', '-rep', type=int, default=6)
    parser.add_argument('--n-reruns', '-rer', type=int, default=5)
    parser.add_argument('--rollout-scheme', '-rs', default='linear', choices=('linear', 'doubling'))
    def parse_lambdas(x): return [float(_) for _ in x.split(',')]
    parser.add_argument('--lambda-grid', '-lg', default=DEFAULT_LAMBDA_GRID, type=parse_lambdas)

    args = parser.parse_args()

    X_train, y_train, X_test, y_test, labels = load_dataset(args.dataset_name)
    pi0, pistar = make_baselines_skylines(args.dataset_name, X_train, y_train, n_jobs=4)

    samples = rollout_indices(len(X_train), args.rollout_scheme, args.n_rollouts)
    print("rollout @", samples)

    crm_loss_by_lambda, crm_reward_by_lambda = run_crm(args, X_train, y_train, X_test, y_test, pi0, samples)

    scrm_loss_by_lambda, scrm_reward_by_lambda = run_scrm(args, X_train, y_train, X_test, y_test, pi0, samples)

    baseline_rewards, skyline_rewards, baseline_loss, skyline_loss = run_baseskyline(args, X_test, y_test, pi0, pistar, samples)

    export_results(args,
                   baseline_loss, skyline_loss, crm_loss_by_lambda, scrm_loss_by_lambda,
                   baseline_rewards, skyline_rewards, crm_reward_by_lambda, scrm_reward_by_lambda)
