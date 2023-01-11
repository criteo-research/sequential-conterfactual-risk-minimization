import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from crm_dataset import CRMDataset
from crm_model import Model
from dataset_utils import load_dataset
from baselines_skylines import make_baselines_skylines, stochastic_hamming_loss
from lambda_heuristic import rollout_indices


def run_crm(X_train, y_train, X_test, y_test, pi0, samples, n_reruns, n_replays,
            scrm: bool = False,
            lambda_: float = .01,
            ips_ix: bool = False, truevar: bool = False):

    crm_losses = np.ones((n_reruns, len(samples)))
    for _ in range(n_reruns):
        np.random.seed(42 + len(X_train) + _*10**4)

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
                # learning
                crm_model.fit(crm_dataset, lambda_=lambda_, sequential_dependence=truevar, ips_ix=ips_ix, snips=not ips_ix)
            # eval
            crm_losses[_, j] = crm_model.expected_hamming_loss(X_test, y_test)
            # next round
            start = end

    # print(crm_losses)
    res = crm_losses.mean(axis=0), crm_losses.std(axis=0)
    print(res[0])
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def parse_datasets(x): return x.split(',')
    parser.add_argument('datasets', type=parse_datasets)
    parser.add_argument('--n-rollouts', '-rol', type=int, default=10)
    parser.add_argument('--n-replays', '-rep', type=int, default=6)
    parser.add_argument('--n-reruns', '-rer', type=int, default=10)
    parser.add_argument('--rollout-scheme', '-rs', default='linear', choices=('linear', 'doubling'))
    def parse_lambdas(x): return [float(_) for _ in x.split(',')]
    parser.add_argument('--lamda', '-lam', default=1e-2, type=float)
    parser.add_argument('--n-jobs', '-j', default=1, type=int)
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

        def _run(l, scrm):
            return run_crm(X_train, y_train, X_test, y_test, pi0, samples,
                           n_reruns=args.n_reruns, n_replays=args.n_replays, lambda_=l, scrm=scrm,
                           truevar=False, ips_ix=args.ips_ix and scrm)

        crm_loss_evolution_mean, crm_loss_evolution_std = _run(1e-4, False)
        print("CRM done")
        scrm_loss_evolution_mean, scrm_loss_evolution_std = _run(1, True)
        print("SCRM done")

        fig, ax1 = plt.subplots(ncols=1, constrained_layout=True, figsize=(8, 4))
        ax1.set_title(dataset)
        ax1.set_xlabel('Samples')
        ax1.plot(samples, np.ones_like(samples)*stochastic_hamming_loss(pi0, X_test, y_test), '--', label='Baseline', color='gray')
        ax1.plot(samples, crm_loss_evolution_mean, 'o-', color='blue', label='CRM')
        ax1.fill_between(samples,
                         crm_loss_evolution_mean - crm_loss_evolution_std,
                         crm_loss_evolution_mean + crm_loss_evolution_std, alpha=.25, color='blue', )
        ax1.plot(samples, scrm_loss_evolution_mean, 'o-', color='orange', label='SCRM')
        ax1.fill_between(samples,
                         scrm_loss_evolution_mean - scrm_loss_evolution_std,
                         scrm_loss_evolution_mean + scrm_loss_evolution_std, alpha=.25, color='orange')
        ax1.plot(samples, np.ones_like(samples)*stochastic_hamming_loss(pistar, X_test, y_test), 'k--', label='Skyline')
        ax1.set_ylabel('Hamming Loss')
        ax1.legend(loc='best')
        plt.savefig('loss_evolution_crm_scrm-%s-%s-%d.png' % (args.rollout_scheme, dataset, args.n_rollouts))

