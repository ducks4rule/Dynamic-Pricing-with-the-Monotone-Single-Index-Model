import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import scipy.stats as sts
import json
import pickle
import os

from typing import List, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor

import densities as dn
from new_algo_1 import grid_search



def generate_X_data(n_samples: int = 100,
                    n_dim: int = 2,
                    params: List = [-1,1],
                    density: str = 'uniform'
                    ) -> npt.NDArray[np.float64]:
    """
    Generate data for X, which is a matrix of shape (n_dim, n_samples).
    - possible densities: 'uniform', 'gaussian', 'laplace'
    - params: parameters for the density function, e.g., for uniform [low, high],
              for gaussian [mean, std_dev], for laplace [mean, scale]
    """
    if density == 'uniform':
        X = np.random.uniform(low=params[0], high=params[1], size=(n_dim, n_samples))
    elif density == 'gaussian':
        X = np.random.normal(loc=params[0], scale=params[1], size=(n_dim, n_samples))
    elif density == 'laplace':
        X = np.random.laplace(loc=params[0], scale=params[1], size=(n_dim, n_samples))

    else:
        raise ValueError('Density not recognised -- try different density')
    return X

def get_prices(n_samples: int = 100,
                 params: List = [0, 5],
                density: str = 'uniform',
               ) -> npt.NDArray[np.float64]:
    """
    Generate prices for the samples.
        - choose density: 'uniform', 'gaussian', 'laplace'
    """
    if density == 'uniform':
        ps = np.random.uniform(low=params[0], high=params[1], size=n_samples)
    elif density == 'gaussian':
        ps = np.random.normal(loc=params[0], scale=params[1], size=n_samples)
    elif density == 'laplace':
        ps = np.random.laplace(loc=params[0], scale=params[1], size=n_samples)
    else:
        raise ValueError('Density not recognised -- try different density')
    return ps

def get_noise(n_samples: int = 100,
              params: List = [0, 1],
              density: str = 'uniform'
              ) -> Tuple[npt.NDArray[np.float64], Callable[[np.ndarray], np.ndarray]]:
    """
    Generate noise for the samples.
        - choose density: 'uniform', 'gaussian', 'laplace'
        - or the densities used in bracale et al. (2025) & fan et al. (2023) with
            - 'bracale_gaussian', 'bracale_laplace', 'bracale_cauchy' -> support=params
            - 'bracale_hoelder' -> params = [alpha] where 0 < alpha < 1
            - 'fan' -> params = [m, [support_low, support_high]] where m = degree of smoothness
            - 'not hoelder' -> non-Hoelder noise
    """
    if density == 'uniform':
        zs = np.random.uniform(low=params[0], high=params[1], size=n_samples)
        cdf = lambda x: sts.uniform.cdf(x, loc=params[0], scale=params[1] - params[0])
    elif density == 'gaussian':
        zs = np.random.normal(loc=params[0], scale=params[1], size=n_samples)
        cdf = lambda x: sts.norm.cdf(x, loc=params[0], scale=params[1])
    elif density == 'laplace':
        zs = np.random.laplace(loc=params[0], scale=params[1], size=n_samples)
        cdf = lambda x: sts.laplace.cdf(x, loc=params[0], scale=params[1])
    elif density == 'bracale_gaussian':
        zs, cdf = dn.sample_bracale_gaussian_laplace_cauchy(n_samples=n_samples, support=params, density='gaussian')
    elif density == 'bracale_laplace':
        zs, cdf = dn.sample_bracale_gaussian_laplace_cauchy(n_samples=n_samples, support=params, density='laplace')
    elif density == 'bracale_cauchy':
        zs, cdf = dn.sample_bracale_gaussian_laplace_cauchy(n_samples=n_samples, support=params, density='cauchy')
    elif density == 'bracale_hoelder':
        zs, cdf = dn.sample_bracale_hoelder(n_samples=n_samples, alpha=params[0])
    elif density == 'fan':
        zs, cdf = dn.sample_noise_fan(n_samples=n_samples, m=params[0], support=params[1])
    elif density == 'not hoelder' or density == 'not_hoelder':
        # zs, cdf = dn.sample_noise_non_hoelder(n_samples=n_samples, supp='large')
        zs, cdf = dn.sample_noise_non_hoelder(n_samples=n_samples, supp='small')
    else:
        raise ValueError('Density not recognised -- try different density')
    return zs, cdf


def get_v_and_y(X: npt.NDArray[np.float64],
                theta: npt.NDArray[np.float64],
                ps: npt.NDArray[np.float64],
                zs: npt.NDArray[np.float64]
                ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Given X, theta, ps, and zs, compute vs and ys.
    vs = theta.T @ X + zs
    ys = indicator(vs >= ps)
    """
    vs = theta.T @ X + zs
    ys = np.where(vs >= ps, 1, 0).flatten()
    return vs, ys



if __name__ == "__main__":
    what_to_calculate = 'theta'
    # what_to_calculate = 'F_hat'
    d = 3
    n = 10000
    params = [-2, 2]
    # density_X = 'uniform'
    density_X = 'uniform'
    density_ps = 'uniform'
    # density_zs = 'uniform'
    # density_zs = 'bracale_gaussian'
    # density_zs = 'bracale_laplace'
    # density_zs = 'bracale_cauchy'
    # density_zs = 'bracale_hoelder'
    # density_zs = 'fan'
    density_zs = 'not hoelder'
    if density_zs in ['uniform', 'bracale_gaussian', 'bracale_laplace', 'bracale_cauchy', 'not hoelder']:
        params_noise = [-1/2, 1/2]
    if density_zs in ['bracale_gaussian', 'bracale_laplace', 'bracale_cauchy']:
        params_noise = [0, 1]
    elif density_zs == 'bracale_hoelder': 
        params_noise = [1/3] # [1/3, 1/2, 3/4]
    elif density_zs == 'fan':
        params_noise = [6, [-1/2, 1/2]] # [2, [-1/2, 1/2]], [4, [-1/2, 1/2]], [6, [-1/2, 1/2]]

    theta = np.random.uniform(low=-1, high=1, size=(d, 1))
    bounds_grid = [[-1, 1]] * d
    run_depth = 2
    n_points = 10

    # max_workers = os.cpu_count() - 1
    max_workers = 30
    if what_to_calculate == 'theta':
        # save_file = 'data/norms_estimator/theta_F_hat_norms_' + density_zs + '.json'
        save_file = 'data/norms_estimator/theta_F_hat_norms_' + density_zs + '_small' + '.json'
        if density_zs == 'bracale_hoelder':
            save_file = 'data/norms_estimator/theta_F_hat_norms_' + density_zs + '_' + str(params_noise[0]) + '.json'
        if density_zs == 'fan':
            save_file = 'data/norms_estimator/theta_F_hat_norms_' + density_zs + '_' + str(params_noise[0]) + '.json'

    if what_to_calculate == 'F_hat':
        # save_file = 'data/norms_estimator/F_hat_3_' + density_zs + '.pkl'
        save_file = 'data/norms_estimator/F_hat_3_' + density_zs + '_small' + '.pkl'
        if density_zs == 'bracale_hoelder':
            save_file = 'data/norms_estimator/F_hat_3_' + density_zs + '_' + str(params_noise[0]) + '.pkl'
        if density_zs == 'fan':
            save_file = 'data/norms_estimator/F_hat_3_' + density_zs + '_' + str(params_noise[0]) + '.pkl'
    if not os.path.exists('data/norms_estimator'):
        os.makedirs('data/norms_estimator')
    
    # X = generate_X_data(n_samples=n, n_dim=d, params=params, density=density_X)
    # ps = get_prices(n_samples=n, params=[0, 5], density=density_ps)
    # zs, F_0 = get_noise(n_samples=n, params=params_noise, density=density_zs)
    # _, ys = get_v_and_y(X, theta, ps, zs)
    # theta_hat, F_hat = grid_search(X.T, ps, ys, bounds=bounds_grid, n_points=10, run_depth=run_depth, verbose=True)


    if what_to_calculate == 'theta':
        d_list = [2, 3, 5]
        n_list = np.logspace(np.log10(1000), np.log10(60000), num=30, dtype=int).tolist()
    if what_to_calculate == 'F_hat':
        d_list = [3]
        n_list = np.logspace(np.log10(100), np.log10(60000), num=15, dtype=int).tolist()


    output_dict = {}
    for d in d_list:
        bounds_grid = [[-2, 2]] * d
        theta = np.random.uniform(low=-2, high=2, size=(d, 1))

        def single_run(n):
            X = generate_X_data(n_samples=n, n_dim=d, params=params, density=density_X)
            ps = get_prices(n_samples=n, params=[0, 5], density=density_ps)
            zs, F_0 = get_noise(n_samples=n, params=params_noise, density=density_zs)
            _, ys = get_v_and_y(X, theta, ps, zs)
            theta_hat, F_hat = grid_search(X.T, ps, ys, bounds=bounds_grid, n_points=n_points, run_depth=run_depth, verbose=False)
                
            if what_to_calculate == 'theta':
                print(f"d={d}, n={n} -- calculating norms...")
                theta_norm = np.linalg.norm(theta_hat - theta.flatten())
                F_hat_norm = np.linalg.norm(F_hat[1, :] - F_0(F_hat[0, :]))
                return theta_norm, F_hat_norm
            elif what_to_calculate == 'F_hat':
                print(f"d={d}, n={n} -- calculating F_hat...")
                return F_hat

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(single_run, n_list))

        if what_to_calculate == 'theta':
            theta_norms, F_hat_norms = zip(*results)
            output_dict['theta_norms' + str(d)] = list(theta_norms)
            output_dict['F_hat' + str(d)] = list(F_hat_norms)
        elif what_to_calculate == 'F_hat':
            output_dict = results

    if what_to_calculate == 'theta':
        with open(save_file, 'w') as f:
            json.dump(output_dict, f)
            print(f"Saved results to {save_file}")
    elif what_to_calculate == 'F_hat':
        with open(save_file, 'wb') as f:
            pickle.dump(output_dict, f)
            print(f"Saved results to {save_file}")

