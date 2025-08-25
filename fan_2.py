import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar

import utils as ut

# -----------------------------
# Kernels
# -----------------------------
def kernel(u, m=2):
    u = np.asarray(u)
    K2 = (35 / 32) * (1 - u**2) ** 3 * (np.abs(u) <= 1).astype(float)
    if m == 2:
        return K2
    elif m == 4:
        return (27 / 15) * (1 - (11 / 3) * u**2) * K2
    elif m == 6:
        return (297 / 128) * (1 - (26 / 3) * u**2 + 13 * u**4) * K2
    else:
        raise ValueError("Unsupported kernel order m. Use 2, 4, or 6.")

def kernel_derivative(u, m=2):
    u = np.asarray(u)
    indicator = (np.abs(u) < 1).astype(float)
    if m == 2:
        return -35/2 * u * (1 - u**2)**2 * indicator
    elif m == 4:
        return -19/16 * ((11*u**2 - 3) * kernel_derivative(u, 2) + 22 * u * kernel(u, 2))
    elif m == 6:
        return 99/128 * ((39*u**4 - 26*u**2 + 3) * kernel_derivative(u, 2) + 52 * u * (3*u**2 - 1) * kernel(u, 2))
    else:
        raise ValueError("Unsupported kernel order m. Use 2, 4, or 6.")

# -----------------------------
# Nadaraya-Watson estimators
# -----------------------------
def nw_estimators(u, theta, X, Y, P, m):
    # u can be a scalar or a 1D array
    w = P - X @ theta
    n = len(Y)
    b = 3 * n**(-1 / (2 * m + 1))
    u = np.atleast_1d(u)
    U = (w[None, :] - u[:, None]) / b  # shape: (len(u), n)
    K_vals = kernel(U, m)              # shape: (len(u), n)
    Kp_vals = kernel_derivative(U, m)  # shape: (len(u), n)

    h = np.sum(K_vals * Y, axis=1) / (n * b)
    f = np.sum(K_vals, axis=1) / (n * b)
    h1 = -np.sum(Kp_vals * Y, axis=1) / (n * b**2)
    f1 = -np.sum(Kp_vals, axis=1) / (n * b**2)

    F_hat = np.where(f != 0, 1 - h / f, 0.0)
    F_prime_hat = np.zeros_like(f)
    np.divide(h1 * f - h * f1, f**2, out=F_prime_hat, where=f != 0)
    # F_prime_hat = np.where(f != 0, -(h1 * f - h * f1) / (f**2), 0.0)
    if F_hat.shape[0] == 1:
        return F_hat[0], F_prime_hat[0]
    return F_hat, F_prime_hat

# -----------------------------
# Least Squares Estimation
# -----------------------------
def estimate_theta(X, Y, B):
    # return np.linalg.pinv(X.T @ X) @ X.T @ (B * Y)
    return np.linalg.lstsq(X, B * Y, rcond=None)[0]

# -----------------------------
# Inverse of phi
# -----------------------------
def invert_phi(u, phi_vals, u_vals):
    def phi_func(z):
        return np.interp(z, u_vals, phi_vals) + u
    res = root_scalar(phi_func, bracket=[u_vals[0], u_vals[-1]])
    return res.root if res.converged else u  # Return the root if converged, else return u

# -----------------------------
# Main Algorithm
# -----------------------------
def dynamic_pricing_fan(X, v_samples, m=2, l_0=10, B=6.0, num_episodes=5, verbose=True):
    X = X.T
    n, d = X.shape
    p_pred = np.zeros(n)
    inds_expl = []
    t = 0
    d = d - 1  # Exclude the intercept term

    for k in range(0, num_episodes):
        ut.verbose_print(f"Episode {k}/{num_episodes}", verbose=verbose)
        
        l_k = 2 ** (k - 1) * l_0
        a_k = int(np.ceil((l_k * d) ** ((2 * m + 1) / (4 * m - 1))))

        # ---------------- Exploration ----------------
        I_k = np.arange(t, min(t + a_k, l_k + t)).astype(int)
        p_samples = np.random.uniform(0, B, size=len(I_k))
        p_pred[I_k] = p_samples

        Y_expl = np.array(v_samples[I_k] >= p_samples, dtype=int)
        X_expl = X[I_k]

        theta_hat = estimate_theta(X_expl, Y_expl, B)

        # Estimate phi and g on a grid
        u_vals = np.linspace(-B, B, 100)
        F, F_prime = nw_estimators(u_vals, theta_hat, X_expl, Y_expl, p_samples, m)
        # plt.plot(u_vals, F, label='F_hat')
        # plt.plot(u_vals, F_prime, label='F_prime_hat')
        # plt.legend()
        # plt.title(f'Episode {k}, Exploration Phase')
        # plt.show()

        # ---------------- Exploitation ----------------
        E_k = np.arange(I_k[-1] + 1, min(l_k + t, v_samples.shape[0])).astype(int)
        inds_expl.extend(E_k.tolist())
        X_exploit = X[E_k]

        phi_vals = np.where(F_prime != 0, u_vals - (1 - F) / F_prime, u_vals)
        g_vals = np.vectorize(lambda u: u + invert_phi(-u, phi_vals, u_vals))(u_vals)
        w = X_exploit @ theta_hat
        P_exploit = np.clip(np.interp(w, u_vals, g_vals), 0, B)

        p_pred[E_k] = P_exploit

        t = t + l_k  # move to next episode


    return p_pred, inds_expl
