import sys
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxopt
from jax.scipy.special import expit as jexpit


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
        res = np.sqrt(18 * complexity_upper_bound + np.log(3 * n_final_samples))
        return res        
        
    @staticmethod
    def random_model(d, k, seed=None):
        np.random.seed(seed)
        beta = jnp.array(np.random.normal(size=(d, k)))
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


    def crm_loss(self, crm_dataset, 
                 snips = True,
                 lambda_: float = 0,
                 max_per_instance_ips = 5e4,
                 max_per_instance_dynamic_log_ips = 50,
                 verbose: int = 0,
                 min_pred: float = 1e-20,
                 max_log_ips: float = 50,
                 min_per_instance_importance_weights: float = 1e-20,
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
        if verbose > 1: jax.debug.print('\tIPS-R: {} - {}', 
                                    per_instance_importance_weighted_rewards.min(), 
                                    per_instance_importance_weighted_rewards.max())
        # SNIPS
        if snips:
            total_loss = per_instance_importance_weighted_rewards.sum() / per_instance_importance_weights.sum()
        else:
            total_loss = per_instance_importance_weighted_rewards.mean()
        
        # POEM
        if lambda_ != 0:
            total_loss += lambda_ * jnp.sqrt(1e-10 + jnp.var(per_instance_importance_weights) / n)
                
        return total_loss / self.k
    
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
                                         tol=loss_args.get('tol',1e-6))
        solution = optimizer.run(self.beta)
        self.beta = solution.params

        if verbose: 
            print("Optim finished:", solution.state)

        return self
