import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from compare_crm_scrm import run_crm_multiple_times, save_config, rollout_indices, add_common_arguments
from dataset_utils import load_dataset


def create_figure(samples,
                  crm_loss_evolution_mean, crm_loss_evolution_std,
                  scrm_loss_evolution_mean, scrm_loss_evolution_std):
    fig, ax1 = plt.subplots(ncols=1, constrained_layout=True, figsize=(8, 4))
    ax1.set_title(args.dataset)
    ax1.set_xlabel('Samples')
    # ax1.plot(samples, np.ones_like(samples)*stochastic_hamming_loss(pi0, X_test, y_test), '--', label='Baseline', color='gray')
    ax1.plot(samples, crm_loss_evolution_mean, 'o-', color='blue', label='CRM')
    ax1.fill_between(samples,
                     crm_loss_evolution_mean - crm_loss_evolution_std,
                     crm_loss_evolution_mean + crm_loss_evolution_std, alpha=.25, color='blue', )
    ax1.plot(samples, scrm_loss_evolution_mean, 'o-', color='orange', label='SCRM')
    ax1.fill_between(samples,
                     scrm_loss_evolution_mean - scrm_loss_evolution_std,
                     scrm_loss_evolution_mean + scrm_loss_evolution_std, alpha=.25, color='orange')
    # ax1.plot(samples, np.ones_like(samples)*stochastic_hamming_loss(pistar, X_test, y_test), 'k--', label='Skyline')
    ax1.set_ylabel('Hamming Loss')
    ax1.legend(loc='best')
    fn = 'loss_evolution_crm_scrm-%s-%s.png' % (args.prefix, args.dataset)
    plt.savefig(fn)
    print('wrote', fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def parse_datasets(x): return x.split(',')
    parser.add_argument('dataset')
    parser.add_argument('lamdacrm', type=float)
    parser.add_argument('lamdascrm', type=float)
    add_common_arguments(parser)
    args = parser.parse_args()

    print(args)

    results = defaultdict(list)

    print("DATASET:", args.dataset)

    X_train, y_train, X_test, y_test, labels = load_dataset(args.dataset)
    X_all = np.vstack([X_train, X_test])
    y_all = np.vstack([y_train, y_test])

    crm_loss_evolutions = run_crm_multiple_times(X_all, y_all, args.dataset, args,
                                                 args.lamdacrm, scrm=False, trace_loss_evolution=True)
    crm_loss_evolution_mean = np.mean(crm_loss_evolutions, axis=0)
    crm_loss_evolution_std = np.std(crm_loss_evolutions, axis=0)

    scrm_loss_evolutions = run_crm_multiple_times(X_all, y_all, args.dataset, args,
                                                 args.lamdascrm, scrm=True, trace_loss_evolution=True)
    scrm_loss_evolution_mean = np.mean(scrm_loss_evolutions, axis=0)
    scrm_loss_evolution_std = np.std(scrm_loss_evolutions, axis=0)

    samples = rollout_indices(len(X_train), args.rollout_scheme, args.n_rollouts)

    create_figure(samples,
                  crm_loss_evolution_mean, crm_loss_evolution_std,
                  scrm_loss_evolution_mean, scrm_loss_evolution_std)
    save_config(args, args.dataset, 'loss_evolution_crm_scrm',
                {'lambda_crm': args.lamdacrm, 'lambda_scrm': args.lamdascrm})
