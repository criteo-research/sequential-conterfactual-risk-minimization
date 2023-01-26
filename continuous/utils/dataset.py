# Libraries
import os
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeCV

dataset_dico = {
    'advertising': 'Advertising',
    'pricing': 'Pricing'
}
from scipy.stats import norm

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


class Advertising(Dataset):
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
        super(Advertising, self).__init__(**kw)
        self.name = name
        self.mode = mode
        self.dimension = 2
        # self.n_samples = n_samples
        self.start_mean = 2.
        self.start_std = sigma
        self.start_sigma = np.sqrt(np.log(self.start_std ** 2 / self.start_mean ** 2 + 1))
        self.start_mu = np.log(self.start_mean) - self.start_sigma ** 2 / 2
        self.mus = [3, 1, 0.1]
        self.potentials_sigma = 0.5
        self.evaluation_offline = False
        self.test_data = self.sample_data(10000, 0)
        self.logging_scale = 0.3
        self.parameter_scale = 1

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

    @staticmethod
    def logging_policy(action, mu, sigma):
        """ Log-normal distribution PDF policy

        Args:
            action (np.array)
            mu (np.array): parameter of log normal pdf
            sigma (np.array): parameter of log normal pdf
        """
        return np.exp(-(np.log(action) - mu) ** 2 / (2 * sigma ** 2)) / (action * sigma * np.sqrt(2 * np.pi))

    def get_X_y(self, n_samples):

        return datasets.make_moons(n_samples=n_samples, noise=.05, random_state=self.random_seed)


    def generate_data(self, n_samples):
        """ Setup the experiments and creates the data
        """
        features, labels = self.get_X_y(n_samples)
        potentials = self._get_potentials(labels)
        actions = self.rng.lognormal(mean=self.start_mu, sigma=self.start_sigma, size=n_samples)
        losses = self.get_losses_from_actions(potentials, actions)
        propensities = self.logging_policy(actions, self.start_mu, self.start_sigma)
        return actions, features, losses, propensities, potentials

    def get_logging_data(self, n_samples):

        actions, contexts, losses, propensities, _ = self.generate_data(n_samples)
        return actions, contexts, losses, propensities

    def sample_data(self, n_samples, index):
        _, contexts, _, _, potentials = self.generate_data(n_samples)
        return contexts, potentials

    @staticmethod
    def get_losses_from_actions(potentials, actions):
        return - np.maximum(np.where(actions < potentials, actions / potentials, -0.5 * actions + 1 + 0.5 * potentials),
                          -0.1)

    def get_optimal_parameter(self, contextual_modelling):
        features, potentials = self.sample_data(10000, 0)
        pistar_determinist = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])

        if contextual_modelling == 'linear':
            embedding = features
        elif contextual_modelling == 'polynomial':
            quadra_features = np.einsum('ij,ih->ijh', features, features).reshape(features.shape[0], -1)
            embedding = np.hstack([features, quadra_features])
        else:
            return
        pistar_determinist.fit(embedding, potentials)
        return np.concatenate([np.array([pistar_determinist.intercept_]), pistar_determinist.coef_]), pistar_determinist


class Pricing(Dataset):
    """Parent class for Data

    """

    def __init__(self, name, mode='quadratic', **kw):
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
        super(Pricing, self).__init__(**kw)
        self.name = name
        self.dimension = 10
        self.l = 3
        self.start_mean = 2.
        self.mode = mode
        self.a, self.b = self.get_functions(self.mode)
        self.test_data = self.sample_data(10000, 0)
        self.logging_scale = 0.5
        self.parameter_scale = 0.01

    def get_functions(self, mode):
        a = lambda z: 2 * z ** 2
        b = lambda z: 0.6 * z
        return a, b


    def generate_data(self, n_samples):
        """ Setup the experiments and creates the data
        """
        z = self.rng.uniform(low=1, high=2, size=(n_samples, self.dimension))
        z_bar = self._get_potentials(z)
        p = self.rng.normal(loc=z_bar, scale=1)
        losses = self.get_losses_from_actions(z_bar, p)
        propensities = norm(loc=z_bar, scale=1).pdf(p)

        return p, z, losses, propensities, z_bar

    def get_logging_data(self, n_samples):

        actions, contexts, losses, propensities, _ = self.generate_data(n_samples)
        return actions, contexts, losses, propensities

    def sample_data(self, n_samples, index):
        _, contexts, _, _, potentials = self.generate_data(n_samples)
        return contexts, potentials

    def _get_potentials(self, z):
        """
        Args
            z (np.array): group labels

        """
        return np.mean(z[:, :self.l], axis=1)

    def get_losses_from_actions(self, z_bar, actions):
        epsilon_noise = self.rng.normal(loc=np.zeros_like(z_bar), scale=1)
        losses = - (actions * (self.a(z_bar) - self.b(z_bar) * actions) + epsilon_noise)
        return np.minimum(losses, np.zeros_like(losses))

    def get_optimal_parameter(self, contextual_modelling):
        z, z_bar = self.sample_data(10000, 0)
        pistar_determinist = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
        optimal_prices = self.a(z_bar) / (2 * self.b(z_bar))
        if contextual_modelling == 'linear':
            embedding = z
        elif contextual_modelling == 'polynomial':
            quadra_z = np.einsum('ij,ih->ijh', z, z).reshape(z.shape[0], -1)
            embedding = np.hstack([z, quadra_z])
        else:
            return
        pistar_determinist.fit(embedding, optimal_prices)
        return np.concatenate([np.array([pistar_determinist.intercept_]), pistar_determinist.coef_]), pistar_determinist


