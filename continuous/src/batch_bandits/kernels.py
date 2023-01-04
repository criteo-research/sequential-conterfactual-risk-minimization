import numpy as np
import jax
import jax.numpy as jnp
import jaxopt


@jax.jit
def sqeuclidean_distance(x, y):
    return jnp.sum((x - y) ** 2)


# Exponential Kernel
@jax.jit
def exp_kernel(gamma, x, y):
    return jnp.exp(- gamma * jnp.sqrt(sqeuclidean_distance(x, y)))


@jax.jit
def polynomial_kernel(dimension, x, y):
    return (jnp.dot(x, y) + 1) ** dimension

def gram(func, params, x, y):
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(params, x1, y1))(y))(x)


class Kernel:

    def __init__(self, settings):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._param = 0.1

    def gram_matrix(self, states):
        return self._pairwise(states, states)

    def evaluate(self, state1, state2):
        return self._pairwise(state1, state2)

    def _pairwise(self, X1, X2):
        pass


class Exponential(Kernel):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(Exponential, self).__init__(*args)
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._alpha = 10

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        return gram(exp_kernel, self._alpha, X1, X2)


class Polynomial(Kernel):

    def __init__(self, *args):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        super(Polynomial, self).__init__(*args)
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self._dimension = 2

    def _pairwise(self, X1, X2):
        """
        Args:
            X1 (np.ndarray)
            X2 (np.ndarray)
        """
        return gram(polynomial_kernel, self._dimension, X1, X2)