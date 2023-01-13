import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from utils.dataset import get_dataset_by_name
from utils.utils import LossHistory, online_evaluation, start_experiment, get_logging_data, update_past_data, dataset_split
from src.crm.model import Model
from src.crm.estimator import Estimator, optimize

def scrm_myopic_experiment(random_seed, dataset_name, settings, lambda_grid=None):

    dataset = get_dataset_by_name(dataset_name, random_seed)
    start_experiment(random_seed, dataset, 'SCRM Myopic')

    # Model setting
    contextual_modelling = Model(settings['contextual_modelling'], random_seed)
    estimator = Estimator(contextual_modelling, dataset.logging_scale)
    scrm_m_loss_history = LossHistory("SCRM-M")

    if settings['data'] == 'geometrical':
        n_samples = settings['n_0']
    else:
        n_samples = settings['T']//settings['M']

    optimal_theta, pistar_determinist = dataset.get_optimal_parameter(settings['contextual_modelling'])
    optimal_loss = online_evaluation(optimal_theta, contextual_modelling, dataset, random_seed)

    # Logging data
    theta = contextual_modelling.create_start_parameter(dataset)
    logging_data = dataset.get_logging_data(n_samples)
    rng = np.random.RandomState(random_seed)
    actions, contexts, losses, propensities = logging_data
    losses_valid = np.zeros_like(lambda_grid)

    for m in range(settings['M']):

        # Optimization


        init_parameter = jnp.array(theta, dtype='float32')

        if settings['validation']:

            train_logging_data, valid_logging_data = dataset_split(contexts, actions, losses, propensities, random_seed)

            for idx, lbd in enumerate(lambda_grid):
                estimator.lbd = lbd
                optimized_theta, loss_crm = optimize(estimator.objective_function, init_parameter, train_logging_data)
                crm_loss_valid = estimator.evaluate(optimized_theta._value, valid_logging_data)
                losses_valid[idx] = crm_loss_valid

            lbd = lambda_grid[np.argmin(losses_valid)]
            estimator.lbd = lbd
            optimized_theta, loss_crm = optimize(estimator.objective_function, init_parameter, logging_data)

        else:
            estimator.lbd = settings['lambda']
            optimized_theta, loss_crm = optimize(estimator.objective_function, init_parameter, logging_data)

        ### New logging data
        theta = optimized_theta._value
        loss_crm = loss_crm._value

        if settings['data'] == 'geometrical':
            n_samples *= 2

        contexts, potentials = dataset.sample_data(n_samples, m)
        contextual_param = contextual_modelling.get_parameter(theta, contexts)
        actions = rng.normal(contextual_param, dataset.logging_scale, n_samples)
        losses = dataset.get_losses_from_actions(potentials, actions)
        propensities = norm(loc=contextual_param, scale=dataset.logging_scale).pdf(actions)
        logging_data = actions, contexts, losses, propensities

        ## Record
        online_loss = online_evaluation(theta, contextual_modelling, dataset, random_seed)
        # regret = online_loss - optimal_loss
        regret = 0.
        cumulated_losses = np.sum(losses)

        scrm_m_loss_history.update(optimized_theta, online_loss, regret, loss_crm, cumulated_losses, 0., 0., n_samples)
        scrm_m_loss_history.show_last()

    return scrm_m_loss_history
