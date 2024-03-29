# Libraries
import os
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeCV


dataset_dico = {
    'noisycircles': 'PotentialPrediction',
    'noisymoons': 'PotentialPrediction',
    'anisotropic': 'PotentialPrediction',
    'warfarin': 'WarfarinDataset',
    'personalized-pricing': 'PersonalizedPricing'
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

    @staticmethod
    def logging_policy(action, mu, sigma):
        """ Log-normal distribution PDF policy

        Args:
            action (np.array)
            mu (np.array): parameter of log normal pdf
            sigma (np.array): parameter of log normal pdf
        """
        return np.exp(-(np.log(action) - mu) ** 2 / (2 * sigma ** 2)) / (action * sigma * np.sqrt(2 * np.pi))

class PotentialPrediction(Dataset):
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
        super(PotentialPrediction, self).__init__(**kw)
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

        elif self.name == 'anisotropic':
            X, y = datasets.make_blobs(n_samples=n_samples, centers=3,
                                       cluster_std=[[1 / 2, 1], [3 / 2, 1 / 2], [1, 3 / 2]],
                                       random_state=self.random_seed)
            X = np.dot(X, self.rng.randn(2, 2))
            return X, y

        else:
            return


    def sample_logged_data(self, n_samples):
        """ Setup the experiments and creates the data
        """
        features, labels = self.get_X_y(n_samples)
        potentials = self._get_potentials(labels)
        actions = self.rng.lognormal(mean=self.start_mu, sigma=self.start_sigma, size=n_samples)
        losses = self.get_losses_from_actions(potentials, actions)
        propensities = self.logging_policy(actions, self.start_mu, self.start_sigma)
        return actions, features, losses, propensities, potentials

    def sample_data(self, n_samples):
        _, contexts, _, _, potentials = self.sample_logged_data(n_samples)
        return contexts, potentials

    @staticmethod
    def get_losses_from_actions(potentials, actions):
        return - np.maximum(np.where(actions < potentials, actions / potentials, -0.5 * actions + 1 + 0.5 * potentials),
                          -0.1)


class PersonalizedPricing(Dataset):
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
        super(PersonalizedPricing, self).__init__(**kw)
        self.name = name
        self.dimension = 10
        self.l = 3
        # self.n_samples = n_samples
        self.start_mean = 2.
        self.mode = mode
        self.a, self.b = self.get_functions(self.mode)
        self.test_data = self.sample_data(10000)

    def get_functions(self, mode):
        if mode == 'quadratic':
            a = lambda z: 2 * z ** 2
            b = lambda z: 0.6 * z
        elif mode == 'step':
            a = lambda z: np.where((z < 1.5), 5, 6)
            b = lambda z: np.where((z < 1.5), 0.7, 1.2)
        elif mode == 'sigmoid':
            a = lambda z: 1 / (1 + np.exp(z))
            b = lambda z: 2 / (1 + np.exp(z)) + 0.1
        else:
            a = lambda z: 6 * z
            b = lambda z: z
        return a, b


    def sample_logged_data(self, n_samples):
        """ Setup the experiments and creates the data
        """
        z = self.rng.uniform(low=1, high=2, size=(n_samples, self.dimension))
        z_bar = self._get_potentials(z)
        p = self.rng.normal(loc=z_bar, scale=1)
        losses = self.get_losses_from_actions(z_bar, p)
        propensities = norm(loc=z_bar, scale=1).pdf(p)

        return p, z, losses, propensities, z_bar

    def sample_data(self, n_samples):
        _, contexts, _, _, potentials = self.sample_logged_data(n_samples)
        return contexts, potentials

    def _get_potentials(self, z):
        """
        Args
            z (np.array): group labels

        """
        return np.mean(z[:, :self.l], axis=1)

    def get_losses_from_actions(self, z_bar, actions):
        epsilon_noise = self.rng.normal(loc=np.zeros_like(z_bar), scale=1)
        return - (actions * (self.a(z_bar) - self.b(z_bar) * actions) + epsilon_noise)

    def get_optimal_parameter(self):
        z, z_bar = self.sample_data(10000)
        pistar_determinist = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
        optimal_prices = self.a(z_bar) / (2 * self.b(z_bar))
        pistar_determinist.fit(z, optimal_prices)
        return np.concatenate([np.array([pistar_determinist.intercept_]), pistar_determinist.coef_])


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
        self.dimension = 81
        self.train_size = 2 / 3
        self.val_size = 0.5
        self._load_and_setup_data()


    def _load_and_setup_data(self):
        """ Load data from csv file
        """
        file_path = os.path.join(self.path, self.file_name)
        data = np.load(file_path)
        features = data[:, :self.dimension]
        actions = data[:, self.dimension]
        losses = - data[:, self.dimension+1]
        propensities = data[:, self.dimension+2]
        potentials = data[:, self.dimension+3]

        self.mu_dose = np.std(potentials)

        idx = self.rng.permutation(features.shape[0])
        features, actions, losses, propensities, potentials = features[idx], actions[idx], losses[idx], \
                                                             propensities[idx], potentials[idx]

        size = int(features.shape[0] * self.train_size)
        a_train, self.actions_test = actions[:size], actions[size:]
        f_train, self.features_test = features[:size, :], features[size:, :]
        l_train, self.losses_test = losses[:size], losses[size:]
        propensities_train, self.propensities_test = propensities[:size], propensities[size:]
        potentials_train, self.potentials_test = potentials[:size], potentials[size:]

        size = int(features.shape[0] * self.val_size)
        self.actions_train, self.actions_valid = a_train[:size], a_train[size:]
        self.features_train, self.features_valid = f_train[:size, :], f_train[size:, :]
        self.losses_train, self.losses_valid = l_train[:size], l_train[size:]
        self.propensities_train, self.propensities_valid = propensities_train[:size], propensities_train[size:]
        self.potentials_train, self.potentials_valid = potentials_train[:size], potentials_train[size:]

        self.baseline_loss_valid = np.mean(self.losses_valid)
        self.baseline_loss_test = np.mean(self.losses_test)


        self.test_data = self.features_test, self.potentials_test

    def get_logged_data(self):
        """ Setup the experiments and creates the data
        """
        return self.actions_train, self.features_train, self.losses_train, self.propensities_train, self.potentials_train

    def get_losses_from_actions(self, potentials, actions):
        return np.maximum(np.abs(potentials - actions) - 0.1*potentials, 0.)

    def sample_logged_data(self, n_samples):
        """ Setup the experiments and creates the data
        """
        return next(self.data_generator)

    def sample_data(self, n_samples):
        _, contexts, _, _, potentials = self.sample_logged_data(n_samples)
        return contexts, potentials

    def set_data_generator(self, n_samples):
        self.data_generator = self._infinite_generator(n_samples)

    # Convert data into a stream of never-ending data
    def _infinite_generator(self, batch_size):

        actions, features, losses, propensities, potentials = self.actions_train, self.features_train, \
                                                              self.losses_train, self.propensities_train, \
                                                              self.potentials_train
        i = 0  # Type: int
        while True:
            j = i + batch_size
            # If we wrap around the back of the dataset:
            if j >= actions.shape[0]:
                rv = list(range(i, actions.shape[0])) + list(range(0, j - actions.shape[0]))
                yield (actions[rv], features[rv, ...], losses[rv], propensities[rv], potentials[rv])
                i = j - actions.shape[0]
            else:
                yield (actions[i:j], features[i:j, ...], losses[i:j], propensities[i:j], potentials[i:j])
                i = j




