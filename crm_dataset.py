import numpy as np

class CRMDataset(object):
    
    def __init__(self):

        self.propensities_ = []
        self.actions_ = []
        self.rewards_ = []
        self.features_ = []
        
        self.propensities_np = None
        self.actions_np = None
        self.rewards_np = None
        self.features_np = None

        self.check()
        
    def __str__(self):
        if not self.propensities_np:
            return '<CRMDataset>'
        self._generate_numpys()
        return '<CRMDataset propensities:%s actions:%s rewards:%s features:%s>' % (
            self.propensities_np.shape,
            self.actions_np.shape,
            self.rewards_np.shape,
            self.features_np.shape
        )
    
    __repr__ = __str__
        
    def check(self):
        assert len(self.features_) == len(self.propensities_) == len(self.rewards_) == len(self.actions_)
        assert type(self.propensities_) == type(self.actions_) == type(self.rewards_) == type(self.features_) == list        
        
    def __len__(self):
        return len(self.propensities)
        
    def _generate_arrays(self):
        if self.propensities_np is not None and len(self.propensities_np) == len(self.propensities_):
            return        
        self.propensities_np = np.vstack(self.propensities_)
        self.actions_np = np.vstack(self.actions_)
        self.features_np = np.vstack(self.features_)
        self.rewards_np = np.vstack(self.rewards_).reshape((self.features.shape[0],1))
        
    @property
    def actions(self):
        return self.actions_np

    @property
    def propensities(self):
        return self.propensities_np
    
    @property
    def rewards(self):
        return self.rewards_np
    
    @property
    def features(self):
        return self.features_np
        
    def update_from_supervised_dataset(self, X, y, probas, n_samples=4):
        # X is (n,d)
        # y is (n,k)
        # probas is (n,k)
        n = X.shape[0]
        d = X.shape[1]
        k = y.shape[1]

        assert len(X) == len(y) == len(probas), (len(X) , len(y) , len(probas))
        
        for _ in range(n_samples):
            actions = (np.random.uniform(size=(n, k)) < probas).astype(int)
            assert actions.shape == (n, k)
            self.actions_ += [actions]
            zero_chosen = np.where(actions == 0)
            propensities = np.array(probas)
            propensities[zero_chosen] = 1 - probas[zero_chosen]
            assert propensities.shape == (n, k)
            self.propensities_ += [propensities]
            self.features_ += [np.array(X)]
            rewards = (1 - np.logical_xor(actions, y)).sum(axis=1).reshape((n,1))
            assert rewards.shape == (n,1), rewards.shape
            self.rewards_ += [rewards]
                
        self._generate_arrays()
        
        assert self.features.shape[0] == self.actions.shape[0] == self.propensities.shape[0] == self.rewards.shape[0]
        assert self.actions.shape[1] == self.propensities.shape[1]
        assert self.rewards.shape[1] == 1
        
        return self
    
if __name__ == '__main__':
    CRMDataset()