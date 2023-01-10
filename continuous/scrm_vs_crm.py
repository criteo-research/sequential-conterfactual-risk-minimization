import numpy as np
from src.crm.crm import repeated_crm_experiment
from src.crm.scrm import scrm_myopic_experiment
import matplotlib.pyplot as plt

def experiments(experiment, dataset_name, settings, lambda_grid):

    histories = []

    for random_seed in range(10):
        loss_history = experiment(random_seed, dataset_name, settings, lambda_grid)
        histories.append(loss_history)

    return histories


dataset_name = 'warfarin'
settings = {
    'contextual_modelling': 'linear',
    'n_0': 100,
    'M': 10,
    'data': 'linear',
    'T': 2642
}

lambda_grid = np.array([1e-3, 1e-2, 1e-1])

# CRM
crm_histories = experiments(repeated_crm_experiment, dataset_name, settings, lambda_grid)
crm_online_losses = np.array([crm_loss_history.online_loss for crm_loss_history in crm_histories])
mean_crm_online_losses = np.mean(crm_online_losses, axis=0)
std_crm_online_loss = np.std(crm_online_losses, axis=0)

# crm_cumulated_losses = np.array([crm_loss_history.cumulated_loss for crm_loss_history in crm_histories])
# mean_crm_cumulated_losses = np.nanmean(crm_cumulated_losses, axis=0)

# SCRM
scrm_m_histories = experiments(scrm_myopic_experiment, dataset_name, settings, lambda_grid)

scrm_m_online_losses = np.array([scrm_m_loss_history.online_loss for scrm_m_loss_history in scrm_m_histories])
mean_scrm_m_online_losses = np.mean(scrm_m_online_losses, axis=0)
std_scrm_m_losses = np.std(scrm_m_online_losses, axis=0)

# scrm_m_cumulated_losses = np.array([scrm_m_loss_history.cumulated_loss for scrm_m_loss_history in scrm_m_histories])
# mean_scrm_m_cumulated_losses = np.nanmean(scrm_m_cumulated_losses, axis=0)

plt.figure()
plt.title('Loss Evolution')
plt.xlabel('m')
plt.plot(mean_crm_online_losses, '-', label='CRM')

plt.fill_between(np.arange(0, 10), mean_crm_online_losses - std_crm_online_loss, mean_crm_online_losses + std_crm_online_loss,
                 color='blue', alpha=0.2)
plt.plot(mean_scrm_m_online_losses, '-.', label='SCRM')
plt.fill_between(np.arange(0, 10), mean_scrm_m_online_losses - std_scrm_m_losses, mean_scrm_m_online_losses + std_scrm_m_losses,
                 color='orange', alpha=0.2)
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('scrm_vs_crm_{}.pdf'.format(dataset_name))


# plt.xlabel('Rollouts')
# plt.ylabel('Cumulated Reward')
# plt.plot(-mean_crm_cumulated_losses, '--', label='CRM')
# plt.plot(-mean_scrm_m_cumulated_losses, '--', label='SCRM-M')
# plt.legend(loc='upper right')
# plt.yscale('log')
# plt.savefig('test_cumulated_losses.pdf')

