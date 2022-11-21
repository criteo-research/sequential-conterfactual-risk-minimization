import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import MultiLabelBinarizer

def load_dataset(dataset_name, test_size=.25):
    X_train, y_train_ = load_svmlight_file(dataset_name+'_train.svm', multilabel=True)
    X_train = np.array(X_train.todense())
    X_test, y_test_ = load_svmlight_file(dataset_name+'_test.svm', multilabel=True)
    X_test = np.array(X_test.todense())
    
    onehot_labeller = MultiLabelBinarizer()
    y_train = onehot_labeller.fit_transform(y_train_).astype(int)
    y_test = onehot_labeller.transform(y_test_).astype(int)
    
    X_all = np.vstack([X_train, X_test])
    y_all = np.vstack([y_train, y_test])
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=0
    )
    
    if dataset_name == 'tmc2007':
        print("reducing dimension for TMC dataset")
        fh = GaussianRandomProjection(n_components=1000)
        X_train = fh.fit_transform(X_train)
        X_test = fh.transform(X_test)
    
    labels = onehot_labeller.classes_.astype(int)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    
    return X_train, y_train, X_test, y_test, labels
    