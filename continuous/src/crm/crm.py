import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from utils.dataset import get_dataset_by_name
from utils.utils import LossHistory, online_evaluation, start_experiment, get_logging_data, update_past_data
from src.crm.model import Model
from src.crm.estimator import Estimator, optimize

def repeated_crm_experiment(random_seed, dataset_name, settings):
    dataset = get_dataset_by_name(dataset_name, random_seed)

    start_experiment(random_seed, dataset, 'CRM')
    # Model setting
    contextual_modelling = Model(settings['contextual_modelling'], random_seed)
    estimator = Estimator(contextual_modelling, 'conservative', settings['lambda'])
    crm_loss_history = LossHistory("CRM")

    optimal_theta = dataset.get_optimal_parameter(settings['contextual_modelling'])
    optimal_loss, _ = online_evaluation(optimal_theta, contextual_modelling, dataset, random_seed)

    n_samples = settings['n_0']

    # Logging data
    theta = contextual_modelling.create_start_parameter(dataset)
    init_parameter = jnp.array(theta, dtype='float32')
    logging_data = get_logging_data(n_samples, dataset)
    rng = np.random.RandomState(random_seed)

    actions, contexts, losses, propensities = logging_data

    for m in range(settings['M']):
        # Optimization
        args = logging_data
        optimized_theta, loss_crm = optimize(estimator.objective_function, init_parameter, args)

        ### New logging data
        loss_crm = loss_crm._value

        n_samples *= 2
        sampled_contexts, sampled_potentials = dataset.sample_data(n_samples)
        contextual_param = contextual_modelling.get_parameter(theta, sampled_contexts)
        sampled_actions = rng.normal(contextual_param, dataset.logging_scale, n_samples)
        sampled_losses = dataset.get_losses_from_actions(sampled_potentials, sampled_actions)
        sampled_propensities = norm(loc=contextual_param, scale=dataset.logging_scale).pdf(sampled_actions)

        cumulated_losses = np.sum(sampled_losses)

        actions = update_past_data(actions, sampled_actions)
        contexts = np.vstack([contexts, sampled_contexts])
        losses = update_past_data(losses, sampled_losses)
        propensities = update_past_data(propensities, sampled_propensities)

        logging_data = actions, contexts, losses, propensities

        ## Record
        online_loss, _ = online_evaluation(optimized_theta._value, contextual_modelling, dataset, random_seed)
        regret = online_loss - optimal_loss

        crm_loss_history.update(optimized_theta, online_loss, regret, loss_crm, cumulated_losses, n_samples)
        crm_loss_history.show_last()

    return crm_loss_history


