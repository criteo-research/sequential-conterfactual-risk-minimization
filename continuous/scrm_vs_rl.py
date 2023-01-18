import numpy as np
from src.crm.crm import repeated_crm_experiment
from src.crm.scrm import scrm_myopic_experiment
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from env_cont import CustomCallback, CustomEnv
import pickle
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th



def experiments(experiment, dataset_name, settings, lambda_grid):

    histories = []

    for random_seed in range(10):
        loss_history = experiment(random_seed, dataset_name, settings, lambda_grid)
        histories.append(loss_history)

    return histories

def initialize_RL_models(env, initial_batch=100, algo='PPO',\
                                      net_arch=[], lr_const=1):
    if algo=='PPO':
        model = PPO('MlpPolicy', env, verbose=0, n_steps=initial_batch)
    else:
        model = TRPO('MlpPolicy', env, verbose=0, n_steps=initial_batch)
    model.policy = ActorCriticPolicy(observation_space= env.observation_space,\
                                     action_space= env.action_space,\
                                     lr_schedule= lambda x: lr_const,\
                                     net_arch=net_arch,\
                                    )
    return model

def pretrain_RL_models(model, env, initial_batch=100, pretraining=250, algo='PPO',\
                                      hyperparam_loss=1):
    #pretrain
    for i in range(pretraining):
        actions = th.squeeze(model.policy._predict(th.from_numpy(env.X))[0])
        _, log_prob, entropy = model.policy.evaluate_actions(th.from_numpy(env.X), actions)
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)
        MSE = (actions-th.from_numpy(env.actions)).pow(2).mean()
        var_loss = MSE\
        +hyperparam_loss*entropy_loss
        model.policy.optimizer.zero_grad()
        var_loss.backward()
        model.policy.optimizer.step()
    callback = CustomCallback()
    model = model.learn(total_timesteps=initial_batch, callback = callback)#, progress_bar=True)
    res=callback.losses_history
    return model, res[0]['lossM'], entropy_loss.item()

def run_rl(results, dataset_name, settings, seed, rl_algo):
    initial_batch = settings['n_0']
    env = CustomEnv(dataset_name, 20000)

    #fit initial policy
    loss_logging = env.losses.mean()
    print('logging perf:',loss_logging, '\n','*'*10, '\n')
    hyper_parap_grid = [.1, .2, .5, 1, 2, 5, 10]
    best_hyperparam=hyper_parap_grid[0]
    best_abs_error = float('inf')
    best_abs_ent = float('inf')
    net_arch = [1]
    lr_const = .1
    algo='TRPO'
    for hyper in hyper_parap_grid:
        env.reset
        model = initialize_RL_models(env, initial_batch=100, algo=algo,\
                                          net_arch=net_arch, lr_const=lr_const)
        model,b,ent = pretrain_RL_models(model, env, initial_batch=100, pretraining=300, algo=algo,\
                                          hyperparam_loss=hyper)
        with open(env.name+'_'+model.__class__.__name__+'hyper'+str(hyper)+'policy.pickle', 'wb') as f:
                pickle.dump(model.policy, f)
        cur_error = abs(b-loss_logging)
        print(hyper, b, cur_error, 'ent:',ent)
        if cur_error<best_abs_error and abs(ent)>0.1:
            best_hyperparam = hyper
            best_abs_error = cur_error
            best_perf=b
            best_abs_ent = ent
    print('best hyper_param found of value:', best_hyperparam, ' and perf:', best_perf, 'entropy:', best_abs_ent)

    #load best policy a posteriori and train with rollouts
    cur_batch_size = initial_batch
    env.reset
    model = initialize_RL_models(env, initial_batch=100, algo=algo,\
                                          net_arch=net_arch, lr_const=lr_const)
    best_policy_name =env.name+'_'+model.__class__.__name__+'hyper'+str(best_hyperparam)+'policy.pickle'
    file = open(best_policy_name,'rb')
    model.policy = pickle.load(file)
    file.close()
    print('loaded from:',best_policy_name )
    callback = CustomCallback()
    model = model.learn(total_timesteps=initial_batch, callback = callback)
    # res=callback.losses_history
    res=[]
    for i in range(settings['M']):
        print('>batch:',i,'of size:',cur_batch_size)
        if i>0:
            model = initialize_RL_models(env, initial_batch=cur_batch_size, algo=algo,                                              net_arch=net_arch, lr_const=lr_const)
            file = open(dataset_name+'_'+model.__class__.__name__+'policy.pickle','rb')
            model.policy = pickle.load(file)
            file.close()
            print('load from:', dataset_name+'_'+model.__class__.__name__+'policy.pickle')
        # callback = CustomCallback()
        model.learn(total_timesteps=cur_batch_size, callback = callback)
        cur_batch_size*=2
        res+=[callback.losses_history[-1]]
        print(res[-1])
    return res[-1]['lossM']



settings = {
    'contextual_modelling': 'linear',
    'n_0': 100,
    'M': 10,
    'data': 'geometrical',
    'validation': True,
    'lambda':0,
    'seeds':2
}

lambda_grid = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
rl_algos = ['PPO', 'TRPO']


def scrm_vs_rl(results, dataset_name, settings, rl_algos):

    # RL
    rl_perfs=[]
    results['dataset'] += [dataset_name]
    for algo in rl_algos:
        for seed in range(settings['seeds']):
            rl_perfs += [run_rl(results, dataset_name, settings, seed, algo)]
        # print(rl_perfs)
        mean_rl_online_losses = np.mean(rl_perfs)
        std_rl_online_loss = np.std(rl_perfs)
        rl_perf, rl_std = mean_rl_online_losses, std_rl_online_loss
        results[algo] += ['$%.3f \pm %.3f$' % (rl_perf, rl_std)]
    #
    # baseline_losses = np.array([crm_loss_history.losses_baseline for crm_loss_history in crm_histories])
    # mean_baseline_losses = np.mean(baseline_losses, axis=0)
    # std_baseline_loss = np.std(baseline_losses, axis=0)
    #
    # skyline_losses = np.array([crm_loss_history.losses_skyline for crm_loss_history in crm_histories])
    # mean_skyline_losses = np.mean(skyline_losses, axis=0)
    # std_skyline_loss = np.std(skyline_losses, axis=0)

    # SCRM
    # scrm_m_histories = experiments(scrm_myopic_experiment, dataset_name, settings, lambda_grid)
    #
    # scrm_m_online_losses = np.array([scrm_m_loss_history.online_loss for scrm_m_loss_history in scrm_m_histories])
    # mean_scrm_m_online_losses = np.mean(scrm_m_online_losses, axis=0)
    # std_scrm_m_losses = np.std(scrm_m_online_losses, axis=0)
    #
    # plt.figure()
    # plt.title('Loss Evolution')
    # plt.xlabel('m')
    # plt.plot(np.arange(1, settings['M']+1), mean_crm_online_losses, '-', label='CRM')
    #
    # plt.fill_between(np.arange(1, settings['M']+1), mean_crm_online_losses - std_crm_online_loss, mean_crm_online_losses + std_crm_online_loss,
    #                  color='blue', alpha=0.2)
    # plt.plot(np.arange(1, settings['M']+1), mean_scrm_m_online_losses, '-.', label='SCRM')
    # plt.fill_between(np.arange(1, settings['M']+1), mean_scrm_m_online_losses - std_scrm_m_losses, mean_scrm_m_online_losses + std_scrm_m_losses,
    #                  color='orange', alpha=0.2)
    # plt.ylabel('Loss')
    # plt.legend(loc='upper right')
    # plt.savefig('scrm_vs_crm_{}.pdf'.format(dataset_name))
    #
    # baseline_perf, baseline_std = mean_baseline_losses[-1], std_baseline_loss[-1]
    # skyline_perf, skyline_std = mean_skyline_losses[-1], std_skyline_loss[-1]
    # scrm_perf, scrm_std = mean_scrm_m_online_losses[-1], std_scrm_m_losses[-1]


    # results['Baseline'] += ['$%.3f \pm %.3f$' % (baseline_perf, baseline_std)]
    # results['RL'] += ['$%.3f \pm %.3f$' % (rl_perf, rl_std)]
    # results['SCRM'] += ['$%.3f \pm %.3f$' % (scrm_perf, scrm_std)]
    # results['Skyline'] += ['$%.3f \pm %.3f$' % (skyline_perf, skyline_std)]

    return results

results = defaultdict(list)

for dataset_name in ['pricing', 'advertising']:
    results = scrm_vs_rl(results, dataset_name, settings, rl_algos)

df = pd.DataFrame(data=results)
df.to_latex(
    'compare_rl_scrm_continuous.tex', index=False, column_format='r', escape=False
)

print('-' * 80)
print(df)
print('-' * 80)
