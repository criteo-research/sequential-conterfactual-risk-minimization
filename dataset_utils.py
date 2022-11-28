import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.preprocessing import add_dummy_feature

def load_dataset(dataset_name, test_size=.25, seed=0, add_intercept=True, scale=False, reduce_dim: int = None):
    X_train, y_train_ = load_svmlight_file(dataset_name+'_train.svm', multilabel=True)
    X_train = np.array(X_train.todense())
    X_test, y_test_ = load_svmlight_file(dataset_name+'_test.svm', multilabel=True)
    X_test = np.array(X_test.todense())       
    
    onehot_labeller = MultiLabelBinarizer()
    y_train = onehot_labeller.fit_transform(y_train_).astype(int)
    y_test = onehot_labeller.transform(y_test_).astype(int)
    
    X_all = np.vstack([X_train, X_test])
    if add_intercept:
        X_all = add_dummy_feature(X_all)
    y_all = np.vstack([y_train, y_test])
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed
    )
    
    if reduce_dim is None and dataset_name == 'tmc2007':
        reduce_dim = 1000
    if reduce_dim:
        print("reducing dimension for TMC dataset")
        fh = GaussianRandomProjection(n_components=reduce_dim)
        X_train = fh.fit_transform(X_train)
        X_test = fh.transform(X_test)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    labels = onehot_labeller.classes_.astype(int)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    
    return X_train, y_train, X_test, y_test, labels
    