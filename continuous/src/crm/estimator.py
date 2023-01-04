import jax
import jax.numpy as jnp
import jaxopt

logging_mu = 3
logging_scale = 0.3

def pdf(loc, x):
    scale = logging_scale
    return 1 / (scale * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-((x - loc) / scale) ** 2 / 2)


class Estimator():
    def __init__(self, contextual_modelling, mode='conservative', lbd=0.1):
        self.contextual_modelling = contextual_modelling
        self.mode = mode
        bonus = 1 if self.mode == 'conservative' else -1
        self.lbd = bonus * lbd

    def objective_function(self, param, actions, contexts, losses, propensities):
        contextual_param = self.contextual_modelling.get_parameter(param, contexts)
        propensities = jnp.clip(propensities, 1e-5, None)
        importance_weights = pdf(contextual_param, actions) / propensities
        mean = jnp.mean(losses * importance_weights)
        std = jnp.std(losses * importance_weights)
        return mean + self.lbd * std


def optimize(loss_fun, init_parameter, args):
    lbfgsb = jaxopt.ScipyMinimize(fun=loss_fun, method="L-BFGS-B").run(init_parameter, *args)
    lbfgs_sol = lbfgsb.params
    lbfgs_fun_val = lbfgsb.state.fun_val

    return lbfgs_sol, lbfgs_fun_val