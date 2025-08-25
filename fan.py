import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import newton
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from scipy.optimize import bisect

import utils as ut


def estimate_theta(X, y, B):
    model = LinearRegression(fit_intercept=False)
    model.fit(X.T, B * y)
    return model.coef_
    # return np.linalg.lstsq(X.T, B * y, rcond=None)[0]

def gaussian_kernel(u):
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

def gaussian_kernel_derivative(u):
    return -u * gaussian_kernel(u)

def kernel_m(u, m = 2):
    def indicator(u):
        return (np.abs(u) <= 1).astype(float)
    if m == 2:
        return 35/12 * (1 - u**2)**3 * indicator(u)
    elif m == 4:
        return 27/16 * (1 - 11/3 * u**2) * kernel_m(u, 2)
    elif m == 6:
        return 297/128 * (1 - 26/3 * u**2 + 13 * u**4) * kernel_m(u, 2)
    else:
        raise ValueError("m must be one of [2, 4, 6] for the Fan et al. kernel.")

def kernel_m_derivative(u, m = 2):
    def indicator(u):
        return (np.abs(u) <= 1).astype(float)
    if m == 2:
        return -35/2 * u * (1 - u**2)**2 * indicator(u)
    elif m == 4:
        return -19/16 * ((11*u**2 - 3) * kernel_m_derivative(u, 2) + 22 * u * kernel_m(u, 2))
    elif m == 6:
        return 99/128 * ((39*u**4 - 26*u**2 + 3) * kernel_m_derivative(u, 2) + 52 * u * (3*u**2 - 1) * kernel_m(u, 2))
    else:
        raise ValueError("m must be one of [2, 4, 6] for the Fan et al. kernel derivative.")
        


def nadaraya_watson(X, p_t, y_t, theta, bandwidth, kernel=kernel_m):
    w_t = lambda th: p_t - np.dot(X.T, th)
    
    h = lambda u, th: np.sum(kernel((w_t(th) - u) / bandwidth) * y_t) / (len(y_t) * bandwidth)
    f = lambda u, th: np.sum(kernel((w_t(th) - u) / bandwidth)) / (len(y_t) * bandwidth)

    F_hat = lambda u: 1 - h(u, theta) / f(u, theta)
    return F_hat

def nadaraya_watson_derivative(X, p_t, y_t, theta, bandwidth, kernel=kernel_m, kernel_prime=kernel_m_derivative):
    w_t = lambda th: p_t - np.dot(X.T, th)

    h = lambda u, th: np.sum(kernel((w_t(th) - u) / bandwidth) * y_t) / (len(y_t) * bandwidth)
    f = lambda u, th: np.sum(kernel((w_t(th) - u) / bandwidth)) / (len(y_t) * bandwidth)

    h_prime = lambda u, th: np.sum(kernel_prime((w_t(th) - u) / bandwidth) * y_t) / (len(y_t) * bandwidth**2)
    f_prime = lambda u, th: np.sum(kernel_prime((w_t(th) - u) / bandwidth)) / (len(y_t) * bandwidth**2)

    F_prime_hat = lambda u: - (h_prime(u, theta) * f(u, theta) - h(u, theta) * f_prime(u, theta)) / (f(u, theta)**2)
    return F_prime_hat

def nw_estimators(X, p_t, y_t, theta, bandwidth, kernel=kernel_m, kernel_prime=kernel_m_derivative, m=2):
    w_t = p_t - np.dot(X.T, theta)

    def F_hat(u):
        h = np.sum(kernel((w_t - u) / bandwidth, m=m) * y_t) / (len(y_t) * bandwidth)
        f = np.sum(kernel((w_t - u) / bandwidth, m=m)) / (len(y_t) * bandwidth)
        if f != 0:
            return 1 - h / f
        else:
            ut.verbose_print("Warning: f(u) is zero, returning 0 for F_hat", verbose=True)
            return 0.0

    def F_prime_hat(u):
        h = np.sum(kernel((w_t - u) / bandwidth, m=m) * y_t) / (len(y_t) * bandwidth)
        f = np.sum(kernel((w_t - u) / bandwidth, m=m)) / (len(y_t) * bandwidth)

        h_prime = np.sum(kernel_prime((w_t - u) / bandwidth, m=m) * y_t) / (len(y_t) * bandwidth**2)
        f_prime = np.sum(kernel_prime((w_t - u) / bandwidth, m=m)) / (len(y_t) * bandwidth**2)

        # return - (h_prime * f - h * f_prime) / (f**2) if f != 0 else 0.0
        return (h_prime * f - h * f_prime) / (f**2) if f != 0 else 0.0

    return F_hat, F_prime_hat

def dynamic_pricing_fan(X, v_samples, m: int, l_0, num_episodes,
                        kernel=kernel_m, kernel_prime=kernel_m_derivative):
    """
    Dynamic pricing algorithm based on the paper of Fan et al. 2022.
    Input:
    - X: covariates at time t
    - v_samples: samples of the valuation function
    - l_0: length of the first episode
    - num_episodes: number of episodes
    - p_min, p_max: minimum and maximum prices
    """
    assert m in [2, 4, 6]

    B = v_samples.max()
    # B = 6.0
    d = X.shape[0] - 1

    p_pred = np.zeros_like(v_samples)
    inds_expl = []

    for k in range(0, num_episodes):
        print(f"Episode {k}")
        l_k = l_0 * 2**(k-1) # length of episode
        l_k_1 = l_0 * 2**k
        a_k = int(np.min([np.ceil((l_k*d)**((2*m + 1)/(4*m -1))), l_k]))  # length of exploration phase
        I_k = np.arange(l_k, l_k + a_k - 1, dtype=int) # indices of the exploration phase
        E_k = np.arange(l_k + a_k, l_k_1, dtype=int) # indices of the exploitation phase 
        bandwidth_k = 3*a_k**(-1/(2*m + 1)) 

        # ---------------- Exploration -----------------
        p_samples = np.random.uniform(low=0, high=B, size=len(I_k))
        p_pred[I_k] = p_samples
        y_t = np.array(v_samples[I_k] >= p_samples, dtype=int)

        theta_hat = estimate_theta(X[:, I_k], y_t, B)

        # F_hat = nadaraya_watson(X[:, I_k], p_samples, y_t, theta_hat, bandwidth_k, kernel)
        # F_prime_hat = nadaraya_watson_derivative(X[:, I_k], p_samples, y_t, theta_hat, bandwidth_k, kernel, kernel_prime)
        F_hat, F_prime_hat = nw_estimators(X[:, I_k], p_samples, y_t, theta_hat, bandwidth_k, kernel)
        phi = np.vectorize(lambda u: u - (1 - F_hat(u)) / F_prime_hat(u) if F_prime_hat(u) != 0 else u)
        F_hat, F_prime_hat = np.vectorize(F_hat), np.vectorize(F_prime_hat)

        # ---------------- Exploitation ----------------
        inds_expl += list(E_k)
        def precompute_phi(phi, res=1000):
            u_grid = np.linspace(-10, 10, res)  # Adjust range and resolution as needed
            phi_values = phi(u_grid)
            inds = np.isfinite(phi_values)
            phi_values = phi_values[inds]
            u_grid = u_grid[inds]
            min_phi, max_phi = phi_values.min(), phi_values.max()
            phi_inverse = interp1d(phi_values, u_grid, bounds_error=False, fill_value="extrapolate")
            return phi_inverse


        phi_inverse = precompute_phi(phi)
        def find_phi_inverse(phi, theta_dot_x):
            def x_to_y(y):
                return -2 * np.exp(y) / (1 + np.exp(y)) + 1

            objective_x = lambda x: phi(x) + theta_dot_x
            objective_y = lambda y: phi(x_to_y(y)) + theta_dot_x

            result = root_scalar(objective_x, bracket=[-10, 10], method='newton', x0=0.0)
            if result.converged:
                return result.root
            else:
                result = root_scalar(objective_y, bracket=[-10, 10], method='newton', x0=0.0)
                if result.converged:
                    return x_to_y(result.root)
                else:
                    return x_to_y(0.0)
                    
        find_phi_inverse = np.vectorize(find_phi_inverse)


        Theta_X = X[:, E_k].T @ theta_hat
        phi_inv = find_phi_inverse(phi, Theta_X)
        # phi_inv = phi_inverse(-Theta_X)
        g_vals = Theta_X + phi_inv

        p_opt = np.clip(g_vals, 0, B)
        p_pred[E_k] = p_opt

    #     xses = np.linspace(-1, 1, len(I_k))
    #     plt.plot(xses, F_hat(xses), label='F_hat')
    #     plt.plot(xses, F_prime_hat(xses), label='F_prime_hat')
    # plt.legend()
    # plt.show()

    return p_pred, inds_expl
