import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import jax.scipy as jsp
from jax import grad, jacfwd, jacrev

def hessian(f):
    return jacfwd(jacrev(f))

@jax.jit
def inverse_gram_matrix(K):
    return jnp.linalg.inv(K)

class SBPE:

    def __init__(self, settings, kernel):
        """Initializes the class

        Attributes:
            random_seed (int):  random seed for data generation process

        """
        self.rng = np.random.RandomState(123)
        self.reg_lambda = settings['lambda']
        self.kernel = kernel
        self.settings = settings
        self.name = 'SBPE'

    def get_story_data(self):
        return self.past_states, self.past_rewards

    def set_gram_matrix(self):
        K = self.kernel.gram_matrix(self.past_states)
        self.K_matrix_inverse = inverse_gram_matrix(K)

    def instantiate(self, env):
        self.action_anchors = env.get_anchor_points()
        actions, contexts, rewards = env.get_logging_data()
        states = self.get_states(contexts, actions)
        self.past_states = jnp.array(states)
        self.past_rewards = jnp.expand_dims(jnp.array(rewards), axis=1)
        self.set_gram_matrix()

    def pure_exploitations(self, states, K_matrix_inverse, S, past_rewards):
        K_S_s = self.kernel.evaluate(S, states)
        return jnp.squeeze(jnp.dot(K_S_s.T, jnp.dot(K_matrix_inverse, past_rewards)))

    def sample_actions(self, contexts):
        S, rewards = self.get_story_data()
        args = self.K_matrix_inverse, S, rewards
        return self.continuous_inference(contexts, args)

    def continuous_inference(self, contexts, args):
        nb_gradient_steps = 0
        if nb_gradient_steps == 0:
            return self.discrete_inference(contexts, args)
        else:
            def func(action):
                state = self.get_state(contexts, action)
                return self.get_upper_confidence_bound(state, *args)

            a0 = self.discrete_inference(contexts, args)
            max_hessian_eigenvalue = jnp.max(jsp.linalg.eigh(hessian(func)(a0), eigvals_only=True))
            step_size = jnp.nan_to_num(1 / max_hessian_eigenvalue)
            a_t = a0
            for _ in range(nb_gradient_steps):
                gradient = jnp.nan_to_num(grad(func)(a_t))
                a_t -= step_size * gradient
            return a_t

    def get_states(self, contexts, actions):
        batch_size = contexts.shape[0]
        contexts, actions = contexts.reshape((batch_size, -1)), actions.reshape((batch_size, 1))
        return jnp.concatenate([contexts, actions], axis=1)

    def get_exploitations(self, contexts, grid, args):
        return jnp.transpose(jnp.array(
            [self.pure_exploitations(self.get_states(contexts, a * np.ones((contexts.shape[0]))), *args) for a in
             grid]))

    def discrete_inference(self, contexts, args):
        grid = self.action_anchors
        exploitations_all_actions = self.get_exploitations(contexts, grid, args)
        idx = jnp.argmax(exploitations_all_actions, axis=1)
        grid = jnp.array(grid)
        return jnp.array([grid[idx]])

    def update_data_pool(self, contexts, actions, rewards):
        states = self.get_states(contexts, actions)
        rewards = np.expand_dims(rewards, axis=1)
        self.past_states = jnp.concatenate([self.past_states, states])
        self.past_rewards = jnp.concatenate([self.past_rewards, rewards])

    def update_agent(self, contexts, actions, rewards):
        self.update_data_pool(contexts, actions, rewards)
        self.set_gram_matrix()

