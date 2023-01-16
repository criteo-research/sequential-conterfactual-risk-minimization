import gym
from gym import spaces
from gym.spaces import MultiDiscrete, Discrete, Box
# import pygame
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import pickle
from dataset_utils import load_dataset


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, dataset_name):
        super(CustomEnv, self).__init__()
        self.name=dataset_name
        self.current_step = 0
        self.binarize_step = 0
        self.X, self.y, self.X_test, self.y_test, labels = load_dataset(dataset_name, seed=1)
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]
        self.k = self.y.shape[1]
        self.reward_range = (0, self.k)
        self.action_space = MultiDiscrete(2*np.ones(shape=self.k))#Tuple((Discrete(2), Box(0, 1, (self.k,))))
        self.observation_space = spaces.Box(
          low=np.array([self.X.min() -10.]*self.d), high=np.array([self.X.max()+10]*self.d))

    def get_X_train(self):
        return self.X

    def get_X_test(self):
        return self.X_test

    def get_y_test(self):
        return self.y_test

    def step(self, action):
        obs = self._next_observation()
        truth = self.y[self.current_step]
        reward = (1 - np.logical_xor(action, truth)).sum()
        done = True
        self.current_step += 1
        if self.current_step >= (self.n):
            self.current_step = 0
        self.binarize_step=self.current_step %2
        return (np.array(obs), reward, done, {})

    def reset(self, seed=None, options=None):
        return self.X[self.current_step]
    def _get_obs(self):
        obs = self.X[self.current_step]
        return obs
    def _get_info(self):
        return ""
    def _next_observation(self):
        obs = self.X[self.current_step]
        return obs


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.X_test=env.get_X_test()
        self.y_test=env.get_y_test()
        self.EHL_history=[]
        self.env=env
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]


    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
         pass

    def _on_step(self) -> bool:
        return True

    def expected_hamming_loss(self, X, y):
            y_invert = 1 - y
            invert_probas = self.predict_proba(X, y_invert)
            return invert_probas.sum() / (self.k * y.shape[0])

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        Eustache implem:
        def expected_hamming_loss(self, X, y):
            y_invert = 1 - y
            invert_probas = self.predict_proba(X, y_invert)
            return invert_probas.sum() / (self.k * y.shape[0])

        """
        res=[np.abs(self.model.predict(self.X_test)[0]-self.y_test).mean() for _ in range(10)]
        self.EHL_history+=[{'step': self.n_calls, 'EHLm': np.mean(res), 'EHLv': np.var(res)}]

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        with open(self.env.name+'_'+self.model.__class__.__name__+'policy.pickle', 'wb') as f:
            pickle.dump(self.model.policy, f)
        with open(self.env.name+'.pickle', 'wb') as f:
            pickle.dump(self.EHL_history, f)
