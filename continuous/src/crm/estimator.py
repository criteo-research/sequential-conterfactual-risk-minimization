import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from scipy.stats import norm


class Estimator():
    def __init__(self, contextual_modelling, scale):
        self.contextual_modelling = contextual_modelling
        self.lbd = 0.
        self.logging_scale = scale

    def pdf(self, loc, x):
        scale = self.logging_scale
        return 1 / (scale * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-((x - loc) / scale) ** 2 / 2)

    def objective_function(self, param, actions, contexts, losses, propensities):
        contextual_param = self.contextual_modelling.get_parameter(param, contexts)
        propensities = jnp.clip(propensities, 1e-5, None)
        importance_weights = self.pdf(contextual_param, actions) / propensities
        mean = jnp.mean(losses * importance_weights)
        std = jnp.std(losses * importance_weights)
        return mean + self.lbd * std

    def evaluate(self, param, logging_data):
        actions, contexts, losses, propensities = logging_data
        contextual_param = self.contextual_modelling.get_parameter(param, contexts)
        propensities = np.clip(propensities, 1e-5, None)
        importance_weights = self.pdf(contextual_param, actions)/propensities
        return np.mean(losses * importance_weights)


class MixtureEstimator():
    def __init__(self, contextual_modelling, scale):
        self.contextual_modelling = contextual_modelling
        self.lbd = 0.
        self.params = []
        self.rollouts_n_samples = []
        self.logging_scale = scale

    def pdf(self, loc, x):
        scale = self.logging_scale
        return 1 / (scale * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-((x - loc) / scale) ** 2 / 2)

    def get_mixture_logging_propensities(self, all_policy_samples, all_contexts):
        distributions = []

        for param in self.params:
            contextual_param = self.contextual_modelling.get_parameter(param, all_contexts)
            distributions.append(norm(loc=contextual_param, scale=self.logging_scale).pdf)
        pi_t = np.array([distribution(all_policy_samples) for distribution in distributions])
        alpha_t = self.rollouts_n_samples / np.sum(self.rollouts_n_samples)
        mixture_logging_propensities = np.sum(alpha_t * pi_t, axis=0)
        return mixture_logging_propensities

    def objective_function(self, param, actions, contexts, losses, mixture_propensities):
        contextual_param = self.contextual_modelling.get_parameter(param, contexts)
        mixture_propensities = jnp.clip(mixture_propensities, 1e-5, None)
        mixture_importance_weights = self.pdf(contextual_param, actions) / mixture_propensities
        mixture_mean = jnp.mean(losses * mixture_importance_weights)
        mixture_std = jnp.sqrt(jnp.sum(jnp.cov(losses * mixture_importance_weights)))
        return mixture_mean + self.lbd * mixture_std

    def update(self, param, rollout_n_samples):
        self.params.append(param)
        self.rollouts_n_samples = np.concatenate([self.rollouts_n_samples, [[rollout_n_samples]]], axis=0)


def optimize(loss_fun, init_parameter, args):
    lbfgsb = jaxopt.ScipyMinimize(fun=loss_fun, method="L-BFGS-B").run(init_parameter, *args)
    lbfgs_sol = lbfgsb.params
    lbfgs_fun_val = lbfgsb.state.fun_val

    return lbfgs_sol, lbfgs_fun_val