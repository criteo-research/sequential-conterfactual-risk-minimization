import sys
import numpy as np

class LossHistory(object):
    
    def __init__(self, name, X_test, y_test):
        self.name = name
        self.X_test = X_test
        self.y_test = y_test
        self.hamming_loss = []
        self.crm_loss = []
        self.unif_crm_loss = []
        self.betas = []
        self.n_samples = []
        self.n_actions = []
        self.rewards = []
        
    def update(self, model, crm_dataset):
        self.betas += [model.beta]
        self.hamming_loss += [model.expected_hamming_loss(self.X_test, self.y_test)]
        self.crm_loss += [model.crm_loss(crm_dataset)]
        self.n_samples += [len(crm_dataset)]
        self.n_actions += [np.sum(crm_dataset.actions)]
        self.rewards += [np.sum(crm_dataset.rewards)]
        
    def show_last(self):
        print(
            '<', self.name,
            '| Ham. loss: %.5f' % self.hamming_loss[-1], 
            '| CRM loss: %.5f' % self.crm_loss[-1],
            '|beta|=%.2f' % np.sqrt((self.betas[-1]**2).sum()), 
            'n=%d' % self.n_samples[-1],
            '|A|=%d' % self.n_actions[-1],
            '|R|=%d' % self.rewards[-1],
            '>',
            file=sys.stderr
        )