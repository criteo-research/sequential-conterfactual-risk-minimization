# Libraries
import os
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

dataset_dico = {
    'open': 'OpenDataset',
    'noisycircles': 'Synthetic',
    'noisymoons': 'Synthetic',
    'anisotropic': 'Synthetic',
    'gmm': 'Synthetic',
    'varied': 'Synthetic',
    'toy-gmm': 'Synthetic',
    'warfarin': 'WarfarinDataset',
}

def get_dataset_by_name(name, random_seed=42):
    mod = __import__("utils.dataset", fromlist=[dataset_dico[name]])
    return getattr(mod, dataset_dico[name])(name=name, random_seed=random_seed)


class Dataset:
    """Parent class for Data

    """
    __metaclass__ = ABCMeta

    def __init__(self, random_seed=42):
        """Initializes the class

        Attributes:
            size (int): size of the dataset
            random_seed (int): random seed for randomized experiments
            rng (numpy.RandomState): random generator for randomization
            train_size (float): train/test ratio
            val_size (float): train/val ratio
            evaluation_offline (bool): perform evaluation offline only, for synthetic dataset this is False

        Note:
            Setup done in auxiliary private method
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    @staticmethod
    def logging_policy(action, mu, sigma):
        """ Log-normal distribution PDF policy

        Args:
            action (np.array)
            mu (np.array): parameter of log normal pdf
            sigma (np.array): parameter of log normal pdf
        """
        return np.exp(-(np.log(action) - mu) ** 2 / (2 * sigma ** 2)) / (action * sigma * np.sqrt(2 * np.pi))

class Synthetic(Dataset):
    """Parent class for Data

    """

    def __init__(self, name, sigma=1, **kw):
        """Initializes the class

        Attributes:
            name (str): name of the dataset
            n_samples (int): number of samples
            start_mean (float): starting mean of the logging policy
            start_std (float): starting std of the logging policy
            start_sigma (float): starting parameter sigma of the logging policy
            start_mu (float): starting parameter mu of the logging policy
            mus (list): list of means of the potential group labels
            potentials_sigma (float): variance of the potential group labels

        Note:
            Setup done in auxiliary private method
        """
        super(Synthetic, self).__init__(**kw)
        self.name = name
        self.dimension = 2
        # self.n_samples = n_samples
        self.start_mean = 2.
        self.start_std = sigma
        self.start_sigma = np.sqrt(np.log(self.start_std ** 2 / self.start_mean ** 2 + 1))
        self.start_mu = np.log(self.start_mean) - self.start_sigma ** 2 / 2
        self.mus = [3, 1, 0.1]
        self.potentials_sigma = 0.5
        self.evaluation_offline = False
        self.test_data = self.sample_data(10000)

    def _get_potentials(self, y):
        """
        Args
            y (np.array): group labels

        """
        n_samples = y.shape[0]
        groups = [self.rng.normal(loc=mu, scale=self.potentials_sigma, size=n_samples) for mu in self.mus]
        potentials = np.ones_like(y, dtype=np.float64)
        for y_value, group in zip(np.unique(y), groups):
            potentials[y == y_value] = group[y == y_value]

        return np.abs(potentials)

    def get_X_y(self, n_samples):
        if self.name == 'noisycircles':
            return datasets.make_circles(n_samples=n_samples, factor=.5,
                                         noise=.05, random_state=self.random_seed)
        elif self.name == 'noisymoons':
            return datasets.make_moons(n_samples=n_samples, noise=.05, random_state=self.random_seed)
        elif self.name == 'gmm':
            return datasets.make_blobs(n_samples=n_samples, random_state=self.random_seed)
        elif self.name == 'anisotropic':
            X, y = datasets.make_blobs(n_samples=n_samples, centers=3,
                                       cluster_std=[[1 / 2, 1], [3 / 2, 1 / 2], [1, 3 / 2]],
                                       random_state=self.random_seed)
            X = np.dot(X, self.rng.randn(2, 2))
            return X, y
        elif self.name == 'toy-gmm':
            X, y = datasets.make_blobs(centers=2, cluster_std=[[3, 1], [3, 2]],
                                       n_samples=n_samples, random_state=self.random_seed)
            return X, y
        else:
            return datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5],
                                       random_state=self.random_seed)

    def sample_logged_data(self, n_samples):
        """ Setup the experiments and creates the data
        """
        features, labels = self.get_X_y(n_samples)
        potentials = self._get_potentials(labels)
        actions = self.rng.lognormal(mean=self.start_mu, sigma=self.start_sigma, size=n_samples)
        losses = self.get_losses_from_actions(potentials, actions)
        propensities = self.logging_policy(actions, self.start_mu, self.start_sigma)
        return actions, features, losses, propensities, potentials, labels

    def sample_data(self, n_samples):
        _, contexts, _, _, potentials, _ = self.sample_logged_data(n_samples)
        return contexts, potentials

    @staticmethod
    def get_losses_from_actions(potentials, actions):
        return - np.maximum(np.where(actions < potentials, actions / potentials, -0.5 * actions + 1 + 0.5 * potentials),
                          -0.1)

class WarfarinDataset(Dataset):
    """ Warfarin Data

    """

    def __init__(self, name, path='data/', **kw):
        """Initializes the class

        Attributes:
            alpha (np.array): parameters of the log-normal distribution
            sigma (float): variance parameter in log-normal distribution
            outlier_ratio (float): outlier ratio

        Note:
            Other attributed inherited from SyntheticData class
        """
        super(WarfarinDataset, self).__init__(**kw)
        self.path = path
        self.file_name = "warfarin.npy"
        self.name = name
        self.features_length = 81
        self._load_and_setup_data()
        self.train_size = 2 / 3
        self.val_size = 0.5

    def _load_and_setup_data(self):
        """ Load data from csv file
        """
        file_path = os.path.join(self.path, self.file_name)
        data = np.load(file_path)
        features = data[:, :self.features_length]
        actions = data[:, self.features_length]
        losses = - data[:, self.features_length+1]
        pi_logging = data[:, self.features_length+2]
        potentials = data[:, self.features_length+3]

        self.mu_dose = np.std(potentials)

        idx = self.rng.permutation(features.shape[0])
        features, actions, rewards, pi_logging, potentials = features[idx], actions[idx], losses[idx], \
                                                             pi_logging[idx], potentials[idx]

        size = int(features.shape[0] * self.train_size)
        a_train, self.actions_test = actions[:size], actions[size:]
        f_train, self.features_test = features[:size, :], features[size:, :]
        r_train, self.reward_test = rewards[:size], rewards[size:]
        pi_0_train, self.pi_0_test = pi_logging[:size], pi_logging[size:]
        potentials_train, self.potentials_test = potentials[:size], potentials[size:]

        size = int(features.shape[0] * self.val_size)
        self.actions_train, self.actions_valid = a_train[:size], a_train[size:]
        self.features_train, self.features_valid = f_train[:size, :], f_train[size:, :]
        self.reward_train, self.reward_valid = r_train[:size], r_train[size:]
        self.pi_0_train, self.pi_0_valid = pi_0_train[:size], pi_0_train[size:]
        self.potentials_train, self.potentials_valid = potentials_train[:size], potentials_train[size:]

        self.baseline_reward_valid = np.mean(self.reward_valid)
        self.baseline_reward_test = np.mean(self.reward_test)

    def get_potentials_labels(self, mode):
        if mode=='train':
            return self.potentials_train, np.empty_like(self.potentials_train)
        elif mode=='test':
            return self.potentials_test, np.empty_like(self.potentials_test)
        else:
            return self.potentials_valid, np.empty_like(self.potentials_valid)

    def get_losses_from_actions(self, potentials, actions):
        return np.maximum(np.abs(potentials - actions) - 0.1*potentials, 0.)




