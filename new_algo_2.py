import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

import utils as ut

def compute_gcm_left_derivatives_scikit(n, y, increasing=False):
    # n: array-like, counts n_1,...,n_L
    # y: array-like, values y_1,...,y_L
    n = np.asarray(n)
    y = np.asarray(y)
    # The slopes are just y_k, weights are n_k
    ir = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
    # Use dummy x values (indices), since we only care about y and weights
    x = np.arange(len(y))
    d = ir.fit_transform(x, y, sample_weight=n)

    return d

def single_index_stochastic_search(X, v, N: int, n_episodes):
    n, d = X.T.shape
    p = np.random.uniform(low=0, high=5, size=n)
    y = np.array(v >= p, dtype=int)

    # stochastic search
    z_samples = np.random.normal(size=(N, d))
    a_k = z_samples / np.linalg.norm(z_samples, axis=1, keepdims=True)
    a_dot_X = a_k @ X 
    counts_per_row = np.zeros_like(a_dot_X)
    for i, row in enumerate(a_dot_X):
        _, inverse, counts = np.unique(row, return_counts=True, return_inverse=True)
        counts_per_row[i] = counts[inverse]
    y_t = y / counts_per_row
    errors = []
    for i, row in enumerate(a_dot_X):
        d_l = compute_gcm_left_derivatives_scikit(counts_per_row[i,:], row)
        err_k = np.linalg.norm(counts_per_row[i,:] * (y_t[i,:] - d_l), 2)
        errors.append(err_k)
        
    k_min = np.argmin(errors)
    d_min = compute_gcm_left_derivatives_scikit(counts_per_row[k_min,:], a_dot_X[k_min,:])
    alpha_min = a_k[k_min,:]
    print('hi')


    return d_min, alpha_min


def plug_in_linear(X, y, n_episodes):
    X_centered = X - np.mean(X, axis=1).reshape(-1, 1)  # center the data
    lr_model = LinearRegression(fit_intercept=False)
    lr_model.fit(X_centered, y)
    return lr_model.coef_

def single_index_plug_in(X, v, n_episodes):
    X = X.T  # transpose to have shape (n, d)
    n, d = X.shape
    n_1 = n // 2
    n_2 = n - n_1
    I_1 = np.arange(n_1)
    I_2 = np.arange(n_1, n)
    p = np.random.uniform(low=0, high=5, size=n)
    y = np.array(v >= p, dtype=int)

    # estimate alpha using plug-in linear regression
    X_1 = X[I_1, :]
    y_1 = y[I_1]
    alpha = plug_in_linear(X_1, y_1, n_episodes)
    alpha_sc = alpha / np.linalg.norm(alpha)

    # estimate Psi with pava
    X_2 = X[I_2, :]
    y_2 = y[I_2]
    d = compute_gcm_left_derivatives_scikit(y_2, alpha_sc @ X_2.T)
    return d, alpha
