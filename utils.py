import numpy as np
from scipy.optimize import newton, minimize_scalar, root_scalar
import inspect
import subprocess


# --------------------------------------------------------
# Verbose function
def verbose_print(string, verbose=True, terminal=True):
    """
    Print a message to the console or terminal.
    """
    parent = inspect.currentframe().f_back.f_code.co_name
    if verbose:
        if terminal:
            subprocess.run(['echo', '[' + parent + '] ' + string])
        else:
            print('[', parent, '] ', string)






# --------------------------------------------------------
# Utility functions
def ecdf(x):
    """Compute the empirical cumulative distribution function (ECDF) of the input data."""
    n = len(x)
    x_sorted = np.sort(x)
    y = np.arange(1, n + 1) / n
    return x_sorted, y

def epdf(x, ecdf=ecdf):
    """Compute the empirical probability density function (EPDF) of the input data."""
    x_sorted, y = ecdf(x)
    pdf = np.gradient(y, x_sorted)
    return x_sorted, pdf

def find_p_star(theta_0, x_t, F_0, F_prime_0):
    """
    Find the optimal price p* for a given x_t.
    """
    theta_dot_x = np.dot(theta_0, x_t)
    objective = lambda p: p - (1 - F_0(p - theta_dot_x)) / F_prime_0(p - theta_dot_x)
    p_star = newton(objective, x0=0.5, tol=1e-3)
    return p_star

def find_quantile(F_0, confidence_level):
    # Solve F_0(z) = target_prob for z
    result = root_scalar(lambda x: F_0(x) - confidence_level, bracket=[-10, 10]) 
    return result.root



# --------------------------------------------------------
def p_star_oracle(X, theta_0, F_0, p_range = [0, 5], verbose=False):
    """
    Compute the optimal price p* for a given x_t.
    """
    d = X.shape[0] - 1
    num_timesteps = X.shape[1]
    p_star = np.zeros(num_timesteps)
    theta_dot_x = np.dot(theta_0, X)
    objective = lambda p, t: -p*(1 - F_0(p - t))
    for t in range(num_timesteps):
        p_star[t] = minimize_scalar(objective, args=(theta_dot_x[t],), bounds=p_range, method='bounded').x

    verbose_print(f"optimal price computed", verbose)
    return p_star


def empirical_regret(p_star, p_pred, v_samples, verbose=False):
    """
    Compute the empirical regret.
    """
    n = len(p_star)
    y_star = np.array([v_samples >= p_star], dtype=int).ravel()
    y_pred = np.array([v_samples >= p_pred], dtype=int).ravel()

    regret = np.zeros(n)
    for t in range(n):
        regret[t] = np.sum(p_star[t] * y_star[t] - p_pred[t] * y_pred[t])
    regret = np.cumsum(regret)

    verbose_print(f"regret computed", verbose)
    return regret




# --------------------------------------------------------
# PAVA
# =--------------------------------------------------------
def pava(y, x=None, weights=None, increasing=True):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=float)

    # if x is not None:
    #     x = np.asarray(x, dtype=float)
    #     # sort y according to sorted x
    #     sorted_indices = np.argsort(x)
    #     y = y[sorted_indices]
    #     weights = weights[sorted_indices]
        
    if not increasing:
        y = -y
    solution = y.copy()
    weight = weights.copy()
    i = 0
    while i < n - 1:
        if solution[i] > solution[i + 1]:
            total_weight = weight[i] + weight[i + 1]
            avg = (solution[i] * weight[i] + solution[i + 1] * weight[i + 1]) / total_weight
            solution[i] = solution[i + 1] = avg
            weight[i] = weight[i + 1] = total_weight
            j = i
            while j > 0 and solution[j - 1] > solution[j]:
                total_weight = weight[j - 1] + weight[j]
                avg = (solution[j - 1] * weight[j - 1] + solution[j] * weight[j]) / total_weight
                solution[j - 1] = solution[j] = avg
                weight[j - 1] = weight[j] = total_weight
                j -= 1
            i = j
        else:
            i += 1
    if not increasing:
        solution = -solution
    return solution
