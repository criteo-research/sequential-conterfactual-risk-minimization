import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys

import jax
import jax.numpy as jnp
import jaxopt
import pandas as pd

c2 = 0.2
alpha = 1

def loss_function(a, y):
    return c2*(a-y)**2 -alpha

# evenly sampled time at 200ms intervals
thetas = np.arange(0.8, 3., 0.01)

def evaluate_theta(theta, random_seed):
    rng = np.random.RandomState(random_seed)
    size=10000
    actions = rng.normal(loc=theta, scale=0.3, size=size)
    y = rng.normal(loc=1, scale=0.3, size=size)
    loss = np.mean(c2*(actions-y)**2 -alpha)
    return loss

logging_mu = 2.5
logging_scale = 0.3

def get_logging_data(n_samples, random_seed=123):

    rng = np.random.RandomState(random_seed)
    y_0 = rng.normal(loc=1, scale=logging_scale)
    action_samples = rng.normal(loc=logging_mu, scale=logging_scale, size=n_samples)
    losses = loss_function(action_samples, y_0)
    logging_pdf = norm(loc=logging_mu, scale=logging_scale).pdf
    propensities = logging_pdf(action_samples)
    loss = np.mean(losses)

    logging_data = action_samples, losses, propensities

    return logging_data


def pdf(loc, x):
    scale = logging_scale
    return 1/(scale * jnp.sqrt(2*jnp.pi)) * jnp.exp(-((x - loc)/scale)**2/2)

def conservative_loss(param, data):
    logging_samples, logging_losses, logging_propensities = data
    n = logging_losses.shape[0]
    lambd = np.sqrt(18*(np.log(n)))
#     lambd = 0.01
    importance_weights = pdf(param, logging_samples)/logging_propensities
    mean = jnp.mean(logging_losses * importance_weights)
    std = jnp.std(logging_losses * importance_weights)
    return mean + lambd / np.sqrt(n) * std


def optimize(loss_fun, init_parameter, args):
    lower_bounds = -3 * jnp.ones_like(init_parameter)
    upper_bounds = 3 * jnp.ones_like(init_parameter)
    bounds = (lower_bounds, upper_bounds)

    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=loss_fun, method="l-bfgs-b").run(init_parameter, bounds=bounds, data=args)

    lbfgs_sol = lbfgsb.params

    return lbfgs_sol

def repeated_crm_experiment(settings, random_seed=123):
    loss_fun = conservative_loss
    print('-' * 80)
    print('CRM experiment, seed {}'.format(random_seed))

    rng = np.random.RandomState(random_seed)
    repeated_crm_online_losses = []

    M = settings['M']
    n_samples = settings['n_0']
    n_total = n_samples
    mu = logging_mu
    logging_pdf = norm(loc=logging_mu, scale=logging_scale).pdf
    optimized_mus = [mu]
    logging_data = get_logging_data(n_samples, random_seed)
    logging_samples, logging_losses, logging_propensities = logging_data

    for m in range(M):
        print('Rollout {}'.format(m))


        init_parameter = jnp.array(logging_mu, dtype='float32')
        args = logging_data

        optimized_mu = optimize(loss_fun, init_parameter, args)
        optimized_mus.append(optimized_mu)

        mu = optimized_mu._value

        n_samples *= 2
        y_m = rng.normal(loc=1, scale=logging_scale)
        action_samples = rng.normal(loc=logging_mu, scale=logging_scale, size=n_samples)
        losses = loss_function(action_samples, y_m)
        propensities = logging_pdf(action_samples)
        online_loss = evaluate_theta(mu, random_seed)
        repeated_crm_online_losses.append(online_loss)

        logging_samples = np.hstack([logging_samples, action_samples])
        logging_losses = np.hstack([logging_losses, losses])
        logging_propensities = np.hstack([logging_propensities, propensities])
        logging_data = logging_samples, logging_losses, logging_propensities
        n_total += n_samples

    return repeated_crm_online_losses


def sequential_crm_experiment(settings, random_seed=123):
    loss_fun = conservative_loss

    print('-' * 80)
    print('SCRM experiment, seed {}'.format(random_seed))

    M = settings['M']
    n_samples = settings['n_0']

    logging_data = get_logging_data(n_samples, random_seed)

    rng = np.random.RandomState(random_seed)

    mu = logging_mu
    optimized_mus = [mu]
    sequential_crm_online_losses = []

    for m in range(M):
        print('Rollout {}'.format(m))

        init_parameter = jnp.array(mu, dtype='float32')
        args = logging_data

        optimized_mu = optimize(loss_fun, init_parameter, args)
        optimized_mus.append(optimized_mu)

        ### New logging data
        n_samples *= 2
        mu = optimized_mu._value

        logging_samples = rng.normal(loc=mu, scale=logging_scale, size=n_samples)
        y_m = rng.normal(loc=1, scale=logging_scale)
        logging_losses = loss_function(logging_samples, y_m)
        online_loss = evaluate_theta(mu, random_seed)
        sequential_crm_online_losses.append(online_loss)
        logging_pdf = norm(loc=mu, scale=logging_scale).pdf
        logging_propensities = logging_pdf(logging_samples)

        logging_data = logging_samples, logging_losses, logging_propensities


    return sequential_crm_online_losses

settings = {
    'M':22,
    'n_0':100
}

repeated_crm_losses = []
sequential_crm_losses = []

for random_seed in range(10):
    repeated_crm_losses.append(repeated_crm_experiment(settings, random_seed))
    sequential_crm_losses.append(sequential_crm_experiment(settings, random_seed))

repeated_crm_losses = np.array(repeated_crm_losses)
crm_losses = np.mean(repeated_crm_losses, axis=0)
crm_losses_std = np.std(repeated_crm_losses, axis=0)

sequential_crm_losses = np.array(sequential_crm_losses)
scrm_losses = np.mean(sequential_crm_losses, axis=0)
scrm_losses_std = np.std(sequential_crm_losses, axis=0)

data=np.vstack([crm_losses, crm_losses_std, scrm_losses, scrm_losses_std])
df = pd.DataFrame(data.T, columns=['CRM loss', 'CRM std', 'SCRM loss', 'SCRM std'])
df.to_csv('gaussian_example.csv')