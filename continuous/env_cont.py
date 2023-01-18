import gym
from gym import spaces
from gym.spaces import MultiDiscrete, Discrete, Box
from utils.dataset import get_dataset_by_name
# import pygame
import pickle
import numpy as np

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
#     metadata = {'render.modes': ['human']}
#     Everything is stored as batch. one setp_size=one whole pass on the train dataset

    def __init__(self, dataset_name, nb):
        super(CustomEnv, self).__init__()
        self.name=dataset_name
        self.current_step = 0
        self.binarize_step = 0
        self.dataset = get_dataset_by_name(dataset_name)
        self.schedule_rollouts = [nb]#schedule_rollouts
        self.nb_data=nb
        self.batch_id=0
#         self.next_resample = self.schedule_rollouts[self.batch_id]
        self.next_resample = nb
        self.resample_batches(self.next_resample)
        self.d=self.X.shape[1]
        self.action_space = Box(np.array([self.actions.min()-1.]), np.array([self.actions.max()+1.]))
        self.observation_space = Box(
          low=np.array([self.X.min()-1.]*self.d), high=np.array([self.X.max()+1.]*self.d), dtype=np.float64)


    def update_batch_ids(self):
#         print(self.batch_id,len(self.schedule_rollouts))
        if self.batch_id!=(len(self.schedule_rollouts)-1):
            self.batch_id+=1
            self.next_resample = self.schedule_rollouts[self.batch_id]

    def resample_batches(self, n):
#         print(n)
#         self.actions, self.X, self.losses, self.propensities, self.potentials = \
        self.actions, self.X, self.losses, self.propensities, self.potentials = self.dataset.generate_data(n_samples=n)

    def get_dataset(self):
        return self.dataset

    def step(self, action):
#         print(self.next_resample)
        obs = self._next_observation()
        if self.binarize_step == 0:
            loss = np.float64(0.)
            done = False
        else:
            truth = self.potentials[self.current_step]
            loss = ( -self.dataset.get_losses_from_actions(truth, action))
            done = True
#         info = self._get_info()
        self.current_step += 1
        if self.current_step >= self.next_resample:
            self.update_batch_ids()
            self.resample_batches(self.next_resample)
            self.current_step = 0
#             print('new_resample:',self.next_resample)
        self.binarize_step=self.current_step %2
        return np.array(obs), loss.item(), done, {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
#         self.current_step = 0
        return self.X[self.current_step]
    def _get_obs(self):
        obs = self.X[self.current_step]
        return obs
#     def _get_info(self):
#         return ""
    def _next_observation(self):
        obs = self.X[(self.current_step+1)%self.next_resample]
        return (obs)

from stable_baselines3.common.callbacks import BaseCallback
import pickle
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # self.model = model
        # print(self.model)
        # self.env = self.training_env
        # self.dataset = self.env.get_dataset()
        # self.contexts, self.potentials = self.dataset.test_data
        # self.losses_history=[]
#         print(env.name)
#         print()
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
          # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
#         self.n_calls = n_calls  # type: int
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
        """
        This method is called before the first rollout starts.
        """
        self.env0 = self.model.env
        self.dataset = self.env0.env_method('get_dataset')[0]
        self.contexts, self.potentials = self.dataset.test_data
        self.losses_history=[]

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
#         print(self.n_calls)
        return True
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
#         self.model = model
        nb_repet = 10
        potentials = ((self.potentials))#.reshape(-1,1)
        test_perf = np.array([- self.dataset.get_losses_from_actions(potentials,\
                                                    np.squeeze(self.model.predict(self.contexts)[0]))\
                                                      for _ in range(nb_repet)])
        multiple_action_sampled = np.array([np.squeeze(self.model.predict(self.contexts)[0]) \
                                            for _ in range(nb_repet)])
        deterministicity_of_policy = multiple_action_sampled.var(axis=0).mean()
        res_losses_agg_mean = test_perf.mean(axis=0)
        self.losses_history+=[{'step': self.n_calls, \
                               'lossM': -np.mean(res_losses_agg_mean), \
                               'lossV': np.var(res_losses_agg_mean), \
                               'deterministicity': deterministicity_of_policy}]
        with open(self.dataset.name+'_'+self.model.__class__.__name__+'policy.pickle', 'wb') as f:
            pickle.dump(self.model.policy, f)
        with open(self.dataset.name+'_'+self.model.__class__.__name__+'_cont.pickle', 'wb') as f:
            pickle.dump(self.losses_history, f)
