import numpy as np
import torch
import numpy.typing as npt
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from itertools import product
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

import utils as ut
# from spava import spav_pytorch, spav_pytorch_qp

def compute_F_hat(y, increasing=True):
    ir = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
    x = np.arange(len(y))
    F_hat = ir.fit_transform(x, y) # assuming no ties
    return F_hat

def compute_F_hat_torch(x, y, increasing=True):
    if not increasing:
        y = -y
    # F_hat = spav_pytorch(x, y, 
    #     mu=10.0, lam=10.0, lr=0.1, steps=500)
    F_hat = spav_pytorch_qp(y, lam=1.0)
    if not increasing:
        F_hat = -F_hat

    return F_hat



def criterion(F, y):
    return np.sum((1 - y - F) ** 2) / len(F)
    # return np.linalg.norm(1 - y - F)

def criterion_torch(F, y):
    return torch.sum((1 - y - F) ** 2) / len(F)
    
def grid_search(
    X: npt.NDArray[np.float64],  # shape: (n, d)
    p_ts: npt.NDArray[np.float64],  # shape: (n,)
    y: npt.NDArray[np.float64],  # shape: (n,)
    bounds=[[-1, 1], [-1, 1], [-1, 1], [-3, 3]],
    n_points: int = 10,
    run_depth: int = 0,
    verbose: bool = False,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        X: shape (n, d)
        p_ts: shape (n,)
        y: shape (n,)
    Returns:
        theta_min: shape (d,)
        result: shape (2, n)
        grid: shape (n_points**d, d) if return_grid is True
        error_vals: shape (n_points**d,) if return_grid is True
    """
# def grid_search(X, p_ts, y, bounds=[[-1, 1], [-1, 1], [-1, 1], [-3, 3]], n_points=10, run_id=0):
    # Generate a grid of points in the specified bounds
    grid = list(product(*[np.linspace(b[0], b[1], n_points) for b in bounds]))
    grid = [np.array(t) for t in grid]
    error_vals = []
    for theta in grid:
        # c_j = p_ts - theta.T @ X
        c_j = p_ts - X @ theta
        # check for ties in c_j
        if len(np.unique(c_j)) < len(c_j):
            print(f"{len(c_j) - len(np.unique(c_j))} ties found for theta {theta}.")
        sorted_indices = np.argsort(c_j)
        # c_j_sorted = c_j[sorted_indices]
        y_sorted = y[sorted_indices]
        F_hat = compute_F_hat(1 - y_sorted)
        A_k = criterion(F_hat, y_sorted)
        error_vals.append(A_k)

    k_min = np.argmin(error_vals)
    theta_min = grid[k_min]

    if run_depth > 0:
        delta = (bounds[0][1] - bounds[0][0]) / n_points
        new_bounds = [[float(t) - delta, float(t) + delta] for t in theta_min]
        theta_min_refined = grid_search(X, p_ts, y, bounds=new_bounds, n_points=n_points, run_depth=run_depth - 1)
        return theta_min_refined

    # c_min = p_ts - theta_min.T @ X
    c_min = p_ts - X @ theta_min
    sorted_indices = np.argsort(c_min)
    y_sorted = y[sorted_indices]
    F_hat = compute_F_hat(1 - y_sorted)

    ut.verbose_print(f"Run {run_depth}: Found theta_min = {theta_min}, error = {error_vals[k_min]}", verbose=verbose)

    # return theta_min, np.array([c_min[sorted_indices], F_hat]), (grid, error_vals)
    return theta_min, np.array([c_min[sorted_indices], F_hat])

def grad_search(
    X: npt.NDArray[np.float64],  # shape: (n, d)
    p_ts: npt.NDArray[np.float64],  # shape: (n,)
    y: npt.NDArray[np.float64],  # shape: (n,)
    bounds=[-1, 1],
    n_iter: int = 100,
    lr: float = 1e-2,
    optimizer: str = 'adamw',
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        X: shape (n, d)
        p_ts: shape (n,)
        y: shape (n,)
    Returns:
        theta_min: shape (d,)
        result: shape (2, n)
    """
    device = torch.device('cpu')
    X_t = torch.tensor(X, dtype=torch.float64, device=device)
    p_ts_t = torch.tensor(p_ts, dtype=torch.float64, device=device)
    y_t = torch.tensor(y, dtype=torch.float64, device=device)

    d = X.shape[1]
    theta = torch.nn.Parameter(torch.empty(d, dtype=torch.float64, device=device))
    with torch.no_grad():
        theta.uniform_(bounds[0], bounds[1])

    if optimizer == 'adamw':
        opt = torch.optim.AdamW([theta], lr=lr, weight_decay=1e-2)
    else:
        opt = torch.optim.SGD([theta], lr=lr)
    
    def loss_fn(theta):
        c_j = p_ts_t - X_t @ theta
        sorted_indices = torch.argsort(c_j)
        y_sorted = y_t[sorted_indices]
        F_hat = compute_F_hat_torch(c_j[sorted_indices], 1 - y_sorted)
        A_k = criterion_torch(F_hat, y_sorted)
        return A_k

    for i in range(1, n_iter + 1):
        opt.zero_grad()
        loss = loss_fn(theta)
        loss.backward()
        opt.step()

        with torch.no_grad():
            for i in range(d):
                theta[i].clamp_(bounds[0], bounds[1])
        if verbose:
            print(f"Iteration {i}/{n_iter}, Loss: {loss.item()}, Theta: {theta}")

    with torch.no_grad():
        c_min = p_ts_t - X_t @ theta
        sorted_indices = torch.argsort(c_min)
        y_sorted = y_t[sorted_indices]
        F_hat = torch.tensor(compute_F_hat(1 - y_sorted.cpu().numpy()))
        ut.verbose_print(f"Final theta_min = {theta.cpu().numpy()}, loss = {loss.item()}", verbose=verbose)
        return theta.cpu().numpy(), np.array([c_min[sorted_indices].cpu().numpy(), F_hat.cpu().numpy()])


def get_F_val(F_hat: np.ndarray, val):
    val = np.atleast_1d(val)
    idx = np.searchsorted(F_hat[0], val, side='left')
    idx[idx == len(F_hat[0])] = -1
    result = F_hat[1][idx]
    return result[0] if result.size == 1 else result

def get_F_val_smooth(F_hat, val):
    # interp_kind = 'cubic'
    interp_kind = 'linear'
    interpolator = interp1d(F_hat[0], F_hat[1], kind=interp_kind, bounds_error=False, fill_value=(F_hat[1][0], F_hat[1][-1]))
    return interpolator(val)

def argmax_p(F_hat, theta, x_t, p_bounds=[0, 5]):
    theta_dot_x = theta.T @ x_t
    results = []
    for t_d_x in theta_dot_x:
        def objective(p):
            # return -p * (1 - get_F_val(F_hat, p - t_d_x))  # negative for maximization
            return -p * (1 - get_F_val_smooth(F_hat, p - t_d_x))  # smooth version
        result = minimize_scalar(objective, bounds=p_bounds, method='bounded')
        results.append(result.x)
    return np.array(results)

def single_index_profile(X, v_samples, l_0: int, num_episodes: int, prices=None, verbose=True, run_depth=2) -> npt.NDArray[np.float64]:
    if prices is None:
        prices_k = np.random.uniform(low=0, high=5, size=l_0)
    else:
        prices_k = prices[:l_0]

    p_opt = np.zeros(len(v_samples))
    p_opt[:l_0] = prices

    X_k = X[:, :l_0]
    y_k = np.where(prices_k < v_samples[:l_0], 1, 0)

    t = 0
    for k in range(1, num_episodes + 1):
        ut.verbose_print(f"Epoch {k}", verbose=verbose)
        # update theta_hat and F_hat_vals
        theta_hat, F_hat_vals = grid_search(X_k.T, prices_k, y_k, n_points=10, run_depth=run_depth, verbose=False)
        # theta_hat, F_hat_vals = grad_search(X_k.T, prices_k, y_k, n_iter=100, lr=1e-2, verbose=False)

        l_k = l_0 * 2**k
        I_k = np.arange(t, t + l_k, dtype=int)
        t = t + l_k
        X_k = X[:, I_k]
        # propose new prices
        prices_k = argmax_p(F_hat_vals, theta_hat, X_k, p_bounds=[0, 5])
        p_opt[I_k] = prices_k
        
        y_k = np.where(v_samples[I_k] >= prices_k, 1, 0)
        # theta_hat, F_hat_vals = grid_search(X_k.T, prices_k, y_k, n_points=10, run_id=2)

    #     plt.plot(F_hat_vals[0,:], F_hat_vals[1,:], '-o', label=f'Epoch {k}')
    # plt.legend()
    # plt.show()
    ut.verbose_print("Final prices computed", verbose=verbose)
        
    return p_opt
