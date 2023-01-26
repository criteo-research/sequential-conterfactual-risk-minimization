import numpy as np
from src.crm.crm import repeated_crm_experiment
from src.crm.scrm import scrm_myopic_experiment
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd



def offline_validation_experiments(experiment, dataset_name, settings, lambda_grid):

    histories = []

    for random_seed in range(10):
        loss_history = experiment(random_seed, dataset_name, settings, lambda_grid)
        histories.append(loss_history)

    # method
    losses = np.array([history.online_loss for history in histories])
    mean_losses = np.mean(losses, axis=0)
    std_losses = np.std(losses, axis=0)

    # baseline
    baseline_losses = np.array([history.losses_baseline for history in histories])
    mean_baseline_losses = np.mean(baseline_losses, axis=0)
    std_baseline_loss = np.std(baseline_losses, axis=0)

    # skyline
    skyline_losses = np.array([history.losses_skyline for history in histories])
    mean_skyline_losses = np.mean(skyline_losses, axis=0)
    std_skyline_loss = np.std(skyline_losses, axis=0)

    baseline_perf, baseline_std = mean_baseline_losses[-1], std_baseline_loss[-1]
    skyline_perf, skyline_std = mean_skyline_losses[-1], std_skyline_loss[-1]

    return mean_losses, std_losses, baseline_perf, baseline_std, skyline_perf, skyline_std

def a_posteriori_experiments(experiment, dataset_name, settings, lambda_grid):

    losses_mean = None
    losses_std = None
    best_loss = np.inf
    print('-' * 80)
    print('BKCUB experiment, dataset {}'.format(dataset_name))


    for lbd in lambda_grid:
        print('-' * 80)
        print('Lambda {}'.format(lbd))

        histories = []


        for random_seed in range(10):
            # a posteriori selection
            print('-' * 80)
            print('Seed {}'.format(random_seed))
            loss_history = experiment(random_seed, dataset_name, settings, lambda_grid)
            histories.append(loss_history)

        # method
        losses = np.array([history.online_loss for history in histories])
        mean_losses = np.mean(losses, axis=0)
        std_losses = np.std(losses, axis=0)

        # baseline
        baseline_losses = np.array([history.losses_baseline for history in histories])
        mean_baseline_losses = np.mean(baseline_losses, axis=0)
        std_baseline_loss = np.std(baseline_losses, axis=0)

        # skyline
        skyline_losses = np.array([history.losses_skyline for history in histories])
        mean_skyline_losses = np.mean(skyline_losses, axis=0)
        std_skyline_loss = np.std(skyline_losses, axis=0)

        lbd_loss = np.squeeze(mean_losses)[-1]

        if lbd_loss < best_loss:
            best_loss = lbd_loss
            losses_mean = mean_losses
            losses_std = std_losses

    baseline_perf, baseline_std = mean_baseline_losses[-1], std_baseline_loss[-1]
    skyline_perf, skyline_std = mean_skyline_losses[-1], std_skyline_loss[-1]

    return losses_mean, losses_std, baseline_perf, baseline_std, skyline_perf, skyline_std

settings = {
    'contextual_modelling': 'linear',
    'n_0': 100,
    'M': 10,
    'data': 'geometrical',
    'validation': False,
    'lambda':0
}

lambda_grid = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

def scrm_vs_crm(results, dataset_name, settings, lambda_grid):

    # CRM

    crm_losses, crm_losses_std, baseline_perf, baseline_std, skyline_perf, skyline_std = a_posteriori_experiments(repeated_crm_experiment, dataset_name, settings, lambda_grid)

    # SCRM
    scrm_losses, scrm_losses_std, _, _, _, _ = a_posteriori_experiments(scrm_myopic_experiment, dataset_name, settings, lambda_grid)


    rollouts = np.arange(1, settings['M']+1)
    fig, ax1 = plt.subplots(ncols=1, constrained_layout=True, figsize=(4, 3), dpi= 100, facecolor='w', edgecolor='k')
    title ='Pricing' if dataset_name == 'pricing' else 'Advertising'
    ax1.set_title(title)
    ax1.set_xlabel('Rollouts $m$')
    ax1.plot(rollouts, crm_losses, 'o-', color='blue', label='CRM')
    ax1.fill_between(rollouts,
                     crm_losses - crm_losses_std,
                     crm_losses + crm_losses_std, alpha=.25, color='blue', )
    ax1.plot(rollouts, scrm_losses, 'o-', color='orange', label='SCRM')
    ax1.fill_between(rollouts,
                     scrm_losses - scrm_losses_std,
                     scrm_losses + scrm_losses_std, alpha=.25, color='orange')

    ax1.set_ylabel('Loss')
    ax1.legend(loc='best')

    plt.savefig('losses_scrm_vs_crm_{}.pdf'.format(dataset_name), bbox_inches = "tight")

    crm_perf, crm_std = crm_losses[-1], crm_losses_std[-1]
    scrm_perf, scrm_std = scrm_losses[-1], scrm_losses_std[-1]


    results['dataset'] += [dataset_name]
    results['Baseline'] += ['$%.3f \pm %.3f$' % (baseline_perf, baseline_std)]
    results['CRM'] += ['$%.3f \pm %.3f$' % (crm_perf, crm_std)]
    results['SCRM'] += ['$%.3f \pm %.3f$' % (scrm_perf, scrm_std)]
    results['Skyline'] += ['$%.3f \pm %.3f$' % (skyline_perf, skyline_std)]

    return results

results = defaultdict(list)

for dataset_name in ['pricing', 'advertising']:
    results = scrm_vs_crm(results, dataset_name, settings, lambda_grid)

df = pd.DataFrame(data=results)
df.to_latex(
    'compare_losses_crm_scrm_continuous.tex', index=False, column_format='r', escape=False
)

print('-' * 80)
print(df)
print('-' * 80)
