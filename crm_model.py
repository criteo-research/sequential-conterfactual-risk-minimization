import sys
import numpy as np

import jax

from crm_dataset import CRMDataset

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxopt
from jax.scipy.special import expit as jexpit

from joblib import Parallel, delayed


class Model(object):
    
    def __init__(self, beta):
        self.d = beta.shape[0]
        self.k = beta.shape[1]
        self.beta_ = beta
        
    @property
    def beta(self):
        return self.beta_
    
    @beta.setter
    def beta(self, beta):
        # beta is (d, k)
        self.beta_ = beta.reshape(self.d, self.k)
        # self.beta_ = jnp.clip(self.beta_, 1e-50, 1e50)
        
    def theoretical_exploration_bonus(self, n_collected_samples: int, n_final_samples: int):
        assert n_collected_samples <= n_final_samples
        complexity_upper_bound = self.d * np.log(n_collected_samples)
        res = np.sqrt(18 * complexity_upper_bound + np.log(3 * n_final_samples)) / np.sqrt(n_collected_samples)
        return res        
        
    @staticmethod
    def random_model(d, k, seed=None):
        rng = np.random.default_rng(seed)
        beta = jnp.array(rng.normal(size=(d, k)))
        return Model(beta)
    
    @staticmethod
    def null_model(d, k):
        beta = jnp.array(np.zeros((d, k)))
        return Model(beta)

    def predict(self, features):
        wx = jnp.dot(features, self.beta_)
        return (wx > 0).astype(int)
    
    def predict_proba(self, features, actions):
        wx = jnp.dot(features, self.beta_)
        actions_sign = 2 * actions - 1
        return jexpit(actions_sign * wx)
    
    def expected_hamming_loss(self, X, y):
        y_invert = 1 - y
        invert_probas = self.predict_proba(X, y_invert)
        return invert_probas.sum() / (self.k * y.shape[0])

    def check_propensity_overfitting(self, ips_weights):
        std = jnp.std(ips_weights)
        n = ips_weights.shape[0]
        avg = jnp.mean(ips_weights)
        lower_bound = avg - 2.96*std/jnp.sqrt(n)
        upper_bound = avg + 2.96*std/jnp.sqrt(n)
        jax.debug.print("\tIPS weights CI: [{} ; {} ; {}]", lower_bound, avg, upper_bound)
#         if not (lower_bound <= 1 <= upper_bound):
#             jax.debug.print("WARN: propensity overfitting detected")
        return (avg < 2).astype(int)

    def variance_penalty(self, rollout_indices, per_instance_importance_weighted_rewards, sequential_dependence: bool):
        if not sequential_dependence:
            return per_instance_importance_weighted_rewards.std() / \
                jnp.sqrt(len(per_instance_importance_weighted_rewards))
        penalty = 0
        for start1, end1 in rollout_indices:
            for start2, end2 in rollout_indices:
                if start1 > start2:
                    continue
                elif start1 == start2:
                    seqvar = jnp.var(per_instance_importance_weighted_rewards[start1:end1])
                    penalty += seqvar
                else:
                    rollout1_data = per_instance_importance_weighted_rewards[start1:end1]
                    rollout2_data = per_instance_importance_weighted_rewards[start2:end2]
                    paired_rollout1_data = rollout1_data[:len(rollout1_data)][:len(rollout2_data)]
                    paired_rollout2_data = rollout2_data[:len(rollout1_data)][:len(rollout2_data)]
                    cov = jnp.cov(paired_rollout1_data[:,0], paired_rollout2_data[:,0])[0, 1]
                    penalty += 2*len(rollout1_data)*len(rollout2_data)*cov
        # jax.debug.print('p:{}', penalty)
        penalty = jnp.clip(penalty, 0, 10000)
        penalty = jnp.sqrt(penalty / len(per_instance_importance_weighted_rewards)**2) / jnp.sqrt(len(per_instance_importance_weighted_rewards))
        return penalty

    def crm_loss(self, crm_dataset: CRMDataset,
                 snips=True,
                 ips_ix=False,
                 lambda_: float = 0,
                 sequential_dependence: bool = True,
                 max_per_instance_ips=5e4,
                 max_per_instance_dynamic_log_ips=50,
                 verbose: int = 0,
                 min_pred: float = 1e-20,
                 max_log_ips: float = 50,
                 min_per_instance_importance_weights: float = 1e-20,
                 compute_ess=False,
                 **args):
        
        n = crm_dataset.features.shape[0]

        # pi
        predictions = self.predict_proba(crm_dataset.features, crm_dataset.actions)      
        predictions = jnp.clip(predictions, min_pred, 1.)
        if verbose > 1: jax.debug.print('\tpreds: {} [{} ; {}]', 
                                    predictions.shape,
                                    predictions.min(), 
                                    predictions.max())
        per_instance_log_predictions = jnp.log(predictions).sum(axis=1)
        if verbose > 1: jax.debug.print('\tpreds / instance: {} [{} ; {}]', 
                                    per_instance_log_predictions.shape,
                                    jnp.exp(per_instance_log_predictions).min(), 
                                    jnp.exp(per_instance_log_predictions).max())

        # pi0
        if ips_ix:
            alpha = 1/n
            per_instance_log_propensities = jnp.log(crm_dataset.propensities + alpha * predictions).sum(axis=1)
        else:
            per_instance_log_propensities = jnp.log(crm_dataset.propensities).sum(axis=1)
        if verbose > 1: 
            jax.debug.print('\tlog props: {} [{} ; {}]', 
                            per_instance_log_propensities.shape,
                            per_instance_log_propensities.min(),
                            per_instance_log_propensities.max())
            zero_props = (crm_dataset.propensities.min(axis=1) < 1e20).astype(int).sum()
            jax.debug.print('\t~zero props: {} / {}', zero_props, n)
            
        # IPS
        per_instance_log_importance_weights = per_instance_log_predictions - per_instance_log_propensities
        
        defunct = 0
        if verbose > 0: defunct = self.check_propensity_overfitting(jnp.exp(per_instance_log_importance_weights))
        
        # clipping
        per_instance_log_importance_weights = jnp.clip(per_instance_log_importance_weights, -max_log_ips, max_log_ips)
        ips_q10, ips_q90 = jnp.quantile(per_instance_log_importance_weights, 
                                        jnp.array([.1,.9]))        
        M = ips_q90 - ips_q10
        M = jnp.max(jnp.array([max_per_instance_dynamic_log_ips, M]))
        if verbose > 1: 
            jax.debug.print("\tIPS q10/90:  exp({}) / exp({}) = exp({}) = {}", ips_q10, ips_q90, M, jnp.exp(M))
            jax.debug.print('\tlog(IPS): [{} ; {}]', 
                            per_instance_log_importance_weights.min(),
                            per_instance_log_importance_weights.max())
            
        per_instance_importance_weights = jnp.exp(per_instance_log_importance_weights)
        per_instance_importance_weights = jnp.clip(per_instance_importance_weights, min_per_instance_importance_weights, jnp.exp(M))        
        per_instance_importance_weights = jnp.clip(per_instance_importance_weights, min_per_instance_importance_weights, max_per_instance_ips)        

        if verbose > 1: jax.debug.print('\tclipped IPS: {} - {}', 
                                        per_instance_importance_weights.min(),
                                        per_instance_importance_weights.max())
        
        per_instance_importance_weights = per_instance_importance_weights.reshape(
            (per_instance_importance_weights.shape[0], 1)
        )
        
        # reweighting past rewards to make a loss
        per_instance_importance_weighted_rewards = jnp.multiply(
            self.k - crm_dataset.rewards,
            per_instance_importance_weights
        )
        if ips_ix:
            per_instance_importance_weighted_rewards = per_instance_importance_weighted_rewards - 1.
        if verbose > 1: jax.debug.print('\tIPS-R: {} - {}',
                                        per_instance_importance_weighted_rewards.min(),
                                        per_instance_importance_weighted_rewards.max())
        
        # ESS
        if compute_ess:
            squared_importance_weights_sum = per_instance_importance_weights.sum() ** 2
            importance_weights_sum_of_squares = (per_instance_importance_weights ** 2).sum()
            effective_sample_size = squared_importance_weights_sum / importance_weights_sum_of_squares / n
        
        # SNIPS or IPS
        if snips:
            total_loss = per_instance_importance_weighted_rewards.sum() / per_instance_importance_weights.sum()
        else:
            total_loss = per_instance_importance_weighted_rewards.mean()
        
        # POEM
        if lambda_ != 0:
            total_loss += lambda_ * self.variance_penalty(crm_dataset.rollout_indices,
                                                          per_instance_importance_weighted_rewards,
                                                          sequential_dependence)

        result = total_loss / self.k
            
        if compute_ess:
            return result, effective_sample_size

        return result
    
    def fit(self, crm_dataset, verbose: int = 0, beta_start: float = 0,
            **loss_args):
        loss_args['verbose'] = verbose
        
        if beta_start is not None:
            self.beta_ = np.ones(self.beta_.shape) * beta_start
        if verbose:
            jax.debug.print('start loss: {}', self.crm_loss(crm_dataset))
        
        def _loss(beta):
            self.beta = beta
            return self.crm_loss(crm_dataset, **loss_args)

        optimizer = jaxopt.ScipyMinimize(method='L-BFGS-B', fun=_loss, 
                                         maxiter=loss_args.get('maxiter', 1000), 
                                         tol=loss_args.get('tol', 1e-6))
        solution = optimizer.run(self.beta)
        self.beta = solution.params

        if verbose: 
            print("Optim finished:", solution.state)

        return self
    
    DEFAULT_GRID = [1e-3, 1e-2, 1e-1, 0, 1, 1e2, 
                    -1e-3, -1e-2, -1e-1, -1, -1e2]

    @staticmethod
    def autotune_lambda(crm_dataset,
                        d: int, k: int,
                        grid=DEFAULT_GRID, verbose: int = 0,
                        seed: int = 0, shuffle: bool = True, folds: int = 3,
                        n_jobs=5, **loss_args):
        loss_args['verbose'] = verbose

        def eval_one_lambda(lambda_: float, seed: int):
            train_crm_dataset, validation_dataset = crm_dataset.split(seed=seed, shuffle=shuffle)
            m = Model.null_model(d, k).fit(train_crm_dataset, lambda_=lambda_, **loss_args)
            loss = np.asarray([m.crm_loss(validation_dataset, lambda_=0, snips=True)])
            return loss[0]

        losses = []
        for lambda_ in grid:
            mean_loss = np.mean(np.asarray(
                Parallel(n_jobs=n_jobs)(delayed(eval_one_lambda)(lambda_, seed+_*10**4) for _ in range(folds))
            ))
            losses += [mean_loss]

        sorted_results = sorted(zip(losses, grid))
        # jax.debug.print('AT:{}', sorted_results)
        return sorted_results[0][1]

    
class EpsilonGreedyModel(object):
    
    def __init__(self, epsilon, beta):
        self.epsilon = epsilon
        self.model = Model(beta)
        self.uniform_model = Model(np.zeros_like(beta))

    @property
    def d(self):
        return self.model.d

    @property
    def k(self):
        return self.model.k


    def predict_proba(self, features, actions, randomize=True):
        predictions = self.model.predict_proba(features, actions)
        if randomize:
            uniform_predictions = self.uniform_model.predict_proba(features, actions)
            predictions = (1 - self.epsilon) * predictions + self.epsilon * uniform_predictions
        return predictions
    
    def expected_hamming_loss(self, X, y):
        y_invert = 1 - y
        invert_probas = self.predict_proba(X, y_invert)
        return invert_probas.sum() / (self.model.k * y.shape[0])

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    def crm_loss(self, *args, **kwargs):
        return self.model.crm_loss(*args, **kwargs)
    
    @property
    def beta(self):
        return self.model.beta

    @staticmethod
    def null_model(d, k, epsilon=.05):
        beta = jnp.array(np.zeros((d, k)))
        return EpsilonGreedyModel(epsilon, beta)
