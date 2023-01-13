import numpy as np
from src.crm.crm import repeated_crm_experiment
from src.crm.scrm import scrm_myopic_experiment
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd



def experiments(experiment, dataset_name, settings, lambda_grid):

    histories = []

    for random_seed in range(10):
        loss_history = experiment(random_seed, dataset_name, settings, lambda_grid)
        histories.append(loss_history)

    return histories



settings = {
    'contextual_modelling': 'linear',
    'n_0': 100,
    'M': 10,
    'data': 'geometrical',
    'validation': True,
    'lambda':0
}

lambda_grid = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])


def scrm_vs_crm(results, dataset_name, settings, lambda_grid):

    # CRM
    crm_histories = experiments(repeated_crm_experiment, dataset_name, settings, lambda_grid)
    crm_online_losses = np.array([crm_loss_history.online_loss for crm_loss_history in crm_histories])
    mean_crm_online_losses = np.mean(crm_online_losses, axis=0)
    std_crm_online_loss = np.std(crm_online_losses, axis=0)

    baseline_losses = np.array([crm_loss_history.losses_baseline for crm_loss_history in crm_histories])
    mean_baseline_losses = np.mean(baseline_losses, axis=0)
    std_baseline_loss = np.std(baseline_losses, axis=0)

    skyline_losses = np.array([crm_loss_history.losses_skyline for crm_loss_history in crm_histories])
    mean_skyline_losses = np.mean(skyline_losses, axis=0)
    std_skyline_loss = np.std(skyline_losses, axis=0)

    # SCRM
    scrm_m_histories = experiments(scrm_myopic_experiment, dataset_name, settings, lambda_grid)

    scrm_m_online_losses = np.array([scrm_m_loss_history.online_loss for scrm_m_loss_history in scrm_m_histories])
    mean_scrm_m_online_losses = np.mean(scrm_m_online_losses, axis=0)
    std_scrm_m_losses = np.std(scrm_m_online_losses, axis=0)

    plt.figure()
    plt.title('Loss Evolution')
    plt.xlabel('m')
    plt.plot(np.arange(1, settings['M']+1), mean_crm_online_losses, '-', label='CRM')

    plt.fill_between(np.arange(1, settings['M']+1), mean_crm_online_losses - std_crm_online_loss, mean_crm_online_losses + std_crm_online_loss,
                     color='blue', alpha=0.2)
    plt.plot(np.arange(1, settings['M']+1), mean_scrm_m_online_losses, '-.', label='SCRM')
    plt.fill_between(np.arange(1, settings['M']+1), mean_scrm_m_online_losses - std_scrm_m_losses, mean_scrm_m_online_losses + std_scrm_m_losses,
                     color='orange', alpha=0.2)
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('scrm_vs_crm_{}.pdf'.format(dataset_name))

    crm_perf, crm_std = mean_crm_online_losses[-1], std_crm_online_loss[-1]
    baseline_perf, baseline_std = mean_baseline_losses[-1], std_baseline_loss[-1]
    skyline_perf, skyline_std = mean_skyline_losses[-1], std_skyline_loss[-1]
    scrm_perf, scrm_std = mean_scrm_m_online_losses[-1], std_scrm_m_losses[-1]


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
    'compare_crm_scrm_continuous.tex', index=False, column_format='r', escape=False
)

print('-' * 80)
print(df)
print('-' * 80)
