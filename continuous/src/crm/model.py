import numpy as np
import jax
import jax.numpy as jnp

class Model(object):

    def __init__(self, name, random_seed):
        self.name = name
        self.rng = np.random.RandomState(random_seed)

    def create_start_parameter(self, dataset):
        d = dataset.dimension
        if self.name == 'linear':
            return np.concatenate([np.array([1.]), self.rng.normal(scale=dataset.parameter_scale, size=d)])

        elif self.name == 'polynomial':
            return self.rng.normal(scale=dataset.parameter_scale, size=d ** 2 + d + 1)
        else:
            return

    def _linear_modelling(self, parameter, features):
        intercept_coeff, mean_coeff = parameter[0], parameter[1:]
        mean = jnp.dot(features, mean_coeff) + intercept_coeff
        return mean

    def _polynomial_modelling(self, parameter, features):
        n = features.shape[1]
        intercept, coeff_lin, coeff_kern = parameter[0], parameter[1:n + 1], parameter[n + 1:]

        m_linear = jnp.dot(features, coeff_lin) + intercept
        f = jnp.einsum('ij,ih->ijh', features, features).reshape(features.shape[0], -1)
        m_kern = jnp.dot(f, coeff_kern)
        mean = m_kern + m_linear
        return mean

    def get_parameter(self, parameter, features):
        if self.name == 'linear':
            return self._linear_modelling(parameter, features)
        elif self.name == 'polynomial':
            return self._polynomial_modelling(parameter, features)
        else:
            return