import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.random_projection import GaussianRandomProjection


def load_warfarin(reduce_dim: int = None, seed:int = 0):

    np.random.seed(seed)  # simulated dosage need to be reproducible

    df = pd.read_csv('warfrin.csv')

    # select patients where stabel dose has been reached & recorded
    df = df[
        ~np.isnan(df['Subject Reached Stable Dose of Warfarin']) &
        ~np.isnan(df['Therapeutic Dose of Warfarin']) &
        ~np.isnan(df['INR on Reported Therapeutic Dose of Warfarin'])
    ]

    # compute BMI and select patients for which it is known
    df['BMI'] = df['Weight (kg)'] * 100 * 100 / df['Height (cm)'] ** 2
    df = df[~np.isnan(df.BMI)]

    categ_feature_indices = [
        3, 5, 7, 8,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 40,
        # could add genotype features
        59, 60, 61, 62, 63, 64, 65, 66
    ]
    continuous_feature_indices = [9, 10, 67]

    categ_encoder = OneHotEncoder(sparse=False)
    X_categ = categ_encoder.fit_transform(df.iloc[:, categ_feature_indices])
    # continuous features are quantized
    conti_encoder = KBinsDiscretizer(n_bins=5)
    X_conti = conti_encoder.fit_transform(df.iloc[:, continuous_feature_indices]).todense()
    X = np.hstack([X_categ, X_conti])

    if reduce_dim:
        print("reducing dimension for Warfarin dataset")
        fh = GaussianRandomProjection(n_components=reduce_dim)
        X = fh.fit_transform(X)

    observed_dose = df['Therapeutic Dose of Warfarin']

    # simulating therapeutic dose
    epsilon = np.random.normal(size=len(X))
    mean_t = np.mean(observed_dose)
    std_t = np.std(observed_dose)
    zbmi = (df.BMI.values - np.mean(df.BMI.values)) / np.std(df.BMI.values)
    action = mean_t + std_t * np.sqrt(.5) * zbmi + epsilon * std_t * np.sqrt(.5)

    # rejecting negative doses
    mask = action > 0
    X = X[mask, :]
    zbmi = zbmi[mask]
    action = action[mask]

    propensity = norm.pdf((action - mean_t - std_t* np.sqrt(.5) * zbmi) / (std_t * np.sqrt(.5)))

    return X, action, propensity