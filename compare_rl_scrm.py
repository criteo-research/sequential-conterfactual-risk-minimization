import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

# from jax.config import config as jax_config
# jax_config.update("jax_debug_nans", True)
import pickle
from joblib import Parallel, delayed

from dataset_utils import load_dataset
from baselines_skylines import make_baselines_skylines, stochastic_hamming_loss
from crm_dataset import CRMDataset
from crm_model import Model
from lambda_heuristic import rollout_indices, DEFAULT_LAMBDA_GRID
from env_discrete import CustomEnv, CustomCallback
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3.common.policies import ActorCriticPolicy

def instance_and_initialize_rl_model(algo, env, batch_length):
    if algo=='PPO':
        model = PPO("MlpPolicy", env, verbose=0, n_steps=batch_length)
    elif algo=='TRPO':
        model = TRPO("MlpPolicy", env, verbose=0, n_steps=batch_length)
    else:
        raise Exception("algo name not implemented")
    model.policy=ActorCriticPolicy(observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule= lambda x :10,
    net_arch= [1])
    return model

def run_rl(dataset, pi0, samples, n_reruns, n_replays, lambda_grid,
            algo: ['PPO', 'TRPO'] = 'PPO', seed=42):
    # lambda_ is None => autotune

    rl_losses = []
    #define env
    env = CustomEnv(dataset)
    #to change to make it robust w.r.t. rollout schedule
    # batch_length=samples[1]
    # nb_rollouts = samples[-1]
    # print(batch_length, nb_rollouts)
    callback = CustomCallback(env)
    # str_to_save_model = 'model_current.zip'
#     for i in range(n_reruns):
    print('seed:',seed)
    np.random.seed(seed * 42 + 1000)
    start=samples[0]
    for i,el in enumerate(samples[1:]):
        end = el
        batch_length=end-start
#         print(i,start, end, batch_length)
        if i==0:
            model = instance_and_initialize_rl_model(algo, env, batch_length)
        else:
            model = instance_and_initialize_rl_model(algo, env, batch_length)
            # model.policy.load(str_to_save_model)
            # print(dataset+'_'+model.__class__.__name__+'policy.pickle')
            file = open(dataset+'_'+model.__class__.__name__+'policy.pickle','rb')
            model.policy = pickle.load(file)
            file.close()
#             print('model loaded from:', str_to_save_model)
        start=end
        print('.', end='')
#         callback = CustomCallback(env)
        model.learn(total_timesteps=batch_length, callback = callback)
        res = callback.EHL_history
        # print(res[-1])
#         model.policy.save(str_to_save_model)
#         print('model saved:', str_to_save_model)
    res = callback.EHL_history
    rl_losses+=[res[-1]['EHLm']]
    return rl_losses

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
    return crm_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def parse_datasets(x): return x.split(',')
    parser.add_argument('datasets', type=parse_datasets)
    parser.add_argument('--n-rollouts', '-rol', type=int, default=10)
    parser.add_argument('--n-replays', '-rep', type=int, default=6)
    parser.add_argument('--n-reruns', '-rer', type=int, default=2)
    parser.add_argument('--rollout-scheme', '-rs', default='linear', choices=('linear', 'doubling'))
    parser.add_argument('--rl-algo', '-rl', default='PPO', choices=('PPO', 'TRPO'))
    def parse_lambdas(x): return [float(_) for _ in x.split(',')]
    parser.add_argument('--lambda-grid', '-lg', default=DEFAULT_LAMBDA_GRID, type=parse_lambdas)
    parser.add_argument('--to', '-to', default='')
    parser.add_argument('--n-jobs', '-j', default=1, type=int)
    parser.add_argument('--truevar', default=False, action='store_true')
    parser.add_argument('--ips-ix', default=False, action='store_true')

    args = parser.parse_args()

    results = defaultdict(list)
    for dataset in args.datasets:
        print("DATASET:", dataset)

        X_train, y_train, X_test, y_test, labels = load_dataset(dataset)
        pi0, pistar = make_baselines_skylines(dataset, X_train, y_train, n_jobs=4)

        samples = rollout_indices(1*len(X_train), args.rollout_scheme, args.n_rollouts)
        print("rollout @", samples)

        def _run(l, algo='SCRM'):
            if algo=='PPO':
                rl_losses = run_rl(dataset, pi0, samples,
                                     n_reruns=args.n_reruns, n_replays=args.n_replays, lambda_grid=args.lambda_grid,
                                     algo=algo, seed=l)
                return np.mean(rl_losses), np.std(rl_losses)
            elif algo=='TRPO':
                rl_losses = run_rl(dataset, pi0, samples,
                                     n_reruns=args.n_reruns, n_replays=args.n_replays, lambda_grid=args.lambda_grid,
                                     algo=algo, seed=l)
                return np.mean(rl_losses), np.std(rl_losses)
            elif algo=='SCRM':
                crm_losses = run_crm(X_train, y_train, X_test, y_test, pi0, samples,
                                     n_reruns=args.n_reruns, n_replays=args.n_replays, lambda_grid=args.lambda_grid,
                                     autotune_lambda=False, lambda_=l, scrm=True,
                                     truevar=False, ips_ix=args.ips_ix and scrm)
                return np.mean(crm_losses), np.std(crm_losses)
        # ppo_losses_stds_rl=[]
        # for j in range(args.n_reruns):
        #     ppo_losses_stds_rl+=[_run(j, 'PPO')]
        ppo_losses_stds_rl = Parallel(n_jobs=args.n_jobs)(delayed(_run)(j, 'PPO') for j in range(args.n_reruns))
        ppo_losses_stds_rl_array=np.array(ppo_losses_stds_rl)
        ppo_best_loss_a_posteriori_mean, ppo_best_loss_a_posteriori_std = tuple(ppo_losses_stds_rl_array.mean(axis=0))
        print(' PPO best loss a posteriori: %.5f +/- %.5f' % (
            ppo_best_loss_a_posteriori_mean, ppo_best_loss_a_posteriori_std)
        )

        # trpo_losses_stds_rl=[]
        # for j in range(args.n_reruns):
        #     trpo_losses_stds_rl+=[_run(j, 'TRPO')]
        trpo_losses_stds_rl = Parallel(n_jobs=args.n_jobs)(delayed(_run)(j, 'TRPO') for j in range(args.n_reruns))
        trpo_losses_stds_rl_array=np.array(trpo_losses_stds_rl)
        trpo_best_loss_a_posteriori_mean, trpo_best_loss_a_posteriori_std = tuple(trpo_losses_stds_rl_array.mean(axis=0))
        print(' TRPO best loss a posteriori: %.5f +/- %.5f' % (
            trpo_best_loss_a_posteriori_mean, trpo_best_loss_a_posteriori_std)
        )

        scrm_losses_stds_scrm = Parallel(n_jobs=args.n_jobs)(delayed(_run)(l, 'SCRM') for l in args.lambda_grid)
        scrm_best_loss_a_posteriori_mean, scrm_best_loss_a_posteriori_std = sorted(scrm_losses_stds_scrm)[0]
        print('SCRM best loss a posteriori: %.5f +/- %.5f' % (
            scrm_best_loss_a_posteriori_mean, scrm_best_loss_a_posteriori_std)
        )

        results['dataset'] += [dataset]
        results['Baseline'] += ['$%.5f$' % stochastic_hamming_loss(pi0, X_test, y_test)]
        results['PPO'] += ['$%.5f \pm %.5f$' % (ppo_best_loss_a_posteriori_mean, ppo_best_loss_a_posteriori_std)]
        results['TRPO'] += ['$%.5f \pm %.5f$' % (trpo_best_loss_a_posteriori_mean, trpo_best_loss_a_posteriori_std)]
        results['SCRM'] += ['$%.5f \pm %.5f$' % (scrm_best_loss_a_posteriori_mean, scrm_best_loss_a_posteriori_std)]
        results['Skyline'] += ['$%.5f$' % stochastic_hamming_loss(pistar, X_test, y_test)]

    df = pd.DataFrame(data=results)
    df.to_latex(
        'compare_crm_scrm_discrete-%s-rs_%s-ro_%d-rr_%d.tex' % (
            dataset, args.rollout_scheme, args.n_rollouts, args.n_reruns
        ), index=False, column_format='r', escape=False
    )
    print('-'*80)
    print(df)
    print('-'*80)
