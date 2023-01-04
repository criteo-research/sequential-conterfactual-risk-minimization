import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from utils.dataset import get_dataset_by_name
from utils.utils import LossHistory, online_evaluation, start_experiment, get_logging_data, update_past_data
from src.crm.model import Model
from src.crm.estimator import Estimator, optimize

def scrm_myopic_experiment(random_seed, dataset_name, settings):

    dataset = get_dataset_by_name(dataset_name, random_seed)
    start_experiment(random_seed, dataset, 'SCRM Myopic')

    # Model setting
    contextual_modelling = Model(settings['contextual_modelling'], random_seed)
    estimator = Estimator(contextual_modelling, 'conservative', settings['lambda'])
    scrm_m_loss_history = LossHistory("SCRM-M")

    n_samples = settings['n_0']
    optimal_theta = dataset.get_optimal_parameter(settings['contextual_modelling'])
    optimal_loss, _ = online_evaluation(optimal_theta, contextual_modelling, dataset, random_seed)

    # Logging data
    theta = contextual_modelling.create_start_parameter(dataset)
    logging_data = get_logging_data(n_samples, dataset)
    rng = np.random.RandomState(random_seed)

    for m in range(settings['M']):

        # Optimization
        init_parameter = jnp.array(theta, dtype='float32')
        args = logging_data
        optimized_theta, loss_crm = optimize(estimator.objective_function, init_parameter, args)

        ### New logging data
        theta = optimized_theta._value
        loss_crm = loss_crm._value

        n_samples *= 2
        contexts, potentials = dataset.sample_data(n_samples)
        contextual_param = contextual_modelling.get_parameter(theta, contexts)
        actions = rng.normal(contextual_param, dataset.logging_scale, n_samples)
        losses = dataset.get_losses_from_actions(potentials, actions)
        propensities = norm(loc=contextual_param, scale=dataset.logging_scale).pdf(actions)
        logging_data = actions, contexts, losses, propensities

        ## Record
        online_loss, _ = online_evaluation(theta, contextual_modelling, dataset, random_seed)
        regret = online_loss - optimal_loss
        cumulated_losses = np.sum(losses)

        scrm_m_loss_history.update(optimized_theta, online_loss, regret, loss_crm, cumulated_losses, n_samples)
        scrm_m_loss_history.show_last()

    return scrm_m_loss_history
