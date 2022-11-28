import numpy as np
import json
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier


crm_paper_results = {
    'scene': {
        'pi0': 1.543, 'ips': 1.193, 'poem': 1.168, 'pi*':.659
    },
    'yeast': {
        'pi0': 5.547, 'ips': 4.635, 'poem': 4.480, 'pi*':2.282

    },
    'tmc2007': {
        'pi0': 3.345, 'ips': 2.808, 'poem': 2.197, 'pi*':1.189        
    },
    'rcv1_topics': {
        'pi0': 1.463, 'ips': 0.921, 'poem': 0.918, 'pi*':0.222        
    }
}

vrcrm_paper_results = {
    'scene': {
        'pi0': 1.887, 'ips': 1.350, 'poem': 1.169, 'pi*':1.423
    },
    'yeast': {
        'pi0': 5.485, 'ips': 4.256, 'poem': 4.480, 'pi*':4.047

    },
    'tmc2007': {
        'pi0': 5.053, 'ips': 4.416, 'poem': 4.505, 'pi*':1.241        
    },
    'rcv1_topics': {
        'pi0': 1.949, 'ips': 1.240, 'poem': 1.306, 'pi*':0.437        
    }
}


def stochastic_hamming_loss(pi, X_test, y_test):
    predictions = pi.predict_proba(X_test)
    predictions = np.array([_[:,1] for _ in predictions]).T
    idx = np.where(y_test == 0)
    fp = predictions[idx].sum()
    idx = np.where(y_test == 1)
    fn = (1-predictions[idx]).sum()
    return (fn+fp)/(y_test.shape[0]*y_test.shape[1])


def result_table(dataset_name, 
                 pi0, pistar, 
                 X_test, y_test, 
                 ips_loss=None, poem_loss=None, dynamic_ips_loss=None):
    
    crm_stats = crm_paper_results[dataset_name]
    vrcrm_stats = vrcrm_paper_results[dataset_name]
    repro_results = json.load(open('repro_results.json'))
    
    if pi0:
        baseline = stochastic_hamming_loss(pi0, X_test, y_test)
    else:
        baseline = repro_results[dataset_name]["Baseline"]
    if pistar:
        skyline = stochastic_hamming_loss(pistar, X_test, y_test)
    else:
        skyline = repro_results[dataset_name]["Skyline"]
    if ips_loss is None:
        ips_loss = repro_results[dataset_name]["IPS"]
    if poem_loss is None:
        poem_loss = repro_results[dataset_name]["POEM"]
    
    print('Baseline -- crm paper: %.3f -- vrcrm paper: %.3f -- ours: %.3f' % (
        crm_stats["pi0"]/y_test.shape[1], 
        vrcrm_stats["pi0"]/y_test.shape[1],         
        baseline
    ))
    print('IPS      -- crm paper: %.3f -- vrcrm paper: %.3f' % (
        crm_stats["ips"]/y_test.shape[1],
        vrcrm_stats["ips"]/y_test.shape[1]        
    ), end='')
    print(' -- ours: %.3f' % ips_loss)
    if dynamic_ips_loss is not None:
        print('Dynamic IPS                                        -- ours: %.3f' % dynamic_ips_loss)
    print('POEM     -- crm paper: %.3f -- vrcrm paper: %.3f' % (
        crm_stats["poem"]/y_test.shape[1],
        vrcrm_stats["poem"]/y_test.shape[1]
    ), end='')
    print(' -- ours: %.3f' % poem_loss)
    print('Skyline  -- crm paper: %.3f -- vrcrm paper: %.3f -- ours: %.3f' % (
        crm_stats["pi*"]/y_test.shape[1], 
        vrcrm_stats["pi*"]/y_test.shape[1], 
        skyline
    ))
    
exploration_bonus = {'scene':.025, 'yeast':2, 'tmc2007':4}
    
def make_baselines_skylines(dataset_name, X_train, y_train, bonus: float = None, mlp=False, n_jobs=4):
    if mlp:
        pistar = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(500, 100, 40, 10,)), n_jobs=n_jobs)
    else:
        pistar = MultiOutputClassifier(LogisticRegressionCV(max_iter=10000, n_jobs=6), n_jobs=n_jobs)
    pistar.fit(X_train, y_train)
    
    n_0 = int(len(X_train)*.05)
    # print('learning pi0 on', n_0, 'data points')
    X_0 = X_train[-n_0:,:]
    y_0 = y_train[-n_0:,:]
    pi0 = MultiOutputClassifier(LogisticRegression(), n_jobs=n_jobs)
    pi0.fit(X_0, y_0)
    
    # making sure every class has non-zero proba and pi0 is not too good
    if bonus is None:
        bonus = exploration_bonus[dataset_name]
    for i in range(y_train.shape[1]):
        pi0.estimators_[i].coef_ = pi0.estimators_[i].coef_ + bonus
        
    return pi0, pistar