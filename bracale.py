import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar


def antitonic_regression(w_t, y_t, support=[-1/2,1/2], epsilon=1e-6):
    """
    Fit an antitonic regression model to the data.
    Input:
    - w_t: covariates
    - y_t: response variable
    Output:
    - model: fitted antitonic regression model
    """
    inds_sorted = np.argsort(w_t)
    w_t = w_t[inds_sorted]
    y_t = y_t[inds_sorted]

    # check for duplicates in w_t
    unique_w_t, indices, counts = np.unique(w_t, return_index=True, return_counts=True)
    o_js = counts
    if len(unique_w_t) < len(w_t):
        # if there are duplicates, we need to average the corresponding y_t values
        y_hat_j = np.array([np.mean(y_t[w_t == w]) for w in unique_w_t])
        w_t = unique_w_t
        print("Duplicates in w_t found and averaged.")
    else:
        y_hat_j = y_t

    m = len(unique_w_t)
    y_hat_rs = np.zeros((m, m))

    # calculating the o_rs
    for r in range(1, m + 1):
        # for s in range(r, m + 1):
        for s in range(r, m):
            
            # Calculate o_rs
            # o_rs = np.sum(o_js[r - 1:s])
            o_rs = np.sum(o_js[r - 1:s+1])

            # Calculate y_hat_rs
            # numerator = np.sum(o_js[r - 1:s] * y_hat_j[r - 1:s])
            numerator = np.sum(o_js[r - 1:s+1] * y_hat_j[r - 1:s+1])
            y_hat_rs[r - 1, s - 1] = numerator / o_rs


    # maximising
    S_0_hat_val = np.zeros(m)
    for j in range(m):
        # Extract the relevant submatrix for r ≤ j and s ≥ j
        submatrix = y_hat_rs[:j + 1, j:]
        # Compute max for each row (over s ≥ j)
        row_max = np.max(submatrix, axis=1)
        # Compute the min over r ≤ j
        S_0_hat_val[j] = np.min(row_max)


    def S_0_hat_get_vals(u, S_vals, input_value):
        u = np.concatenate((u, [support[1]]))
        S_vals = np.concatenate(([1], S_vals))
        # Find the index of the next biggest value in u
        test_j = np.min(np.where(input_value <= u)[0]) if np.where(input_value <= u)[0].size > 0 else -1
        for j, u_j in enumerate(u):
            if input_value <= u_j:
                return S_vals[j]
        # If input_value is bigger than all u[j], return 0
        return 0

    S_0_hat = np.vectorize(lambda x: S_0_hat_get_vals(u, S_0_hat_val, x))

    o = 0

    return S_0_hat, S_0_hat_val



def antitonic_regression_sklearn(w_t, y_t, **kwargs):
    # w_t = np.concatenate([w_t, [np.max(w_t), np.min(w_t)]])
    # y_t = np.concatenate([y_t, [0, 1]])
    iso_regressor = IsotonicRegression(increasing=False)
    iso_regressor.fit(w_t, y_t)
    S_0_hat = lambda x: iso_regressor.predict(np.atleast_1d(x))
    S_0_hat_val = iso_regressor.predict(np.sort(np.unique(w_t)))
    return S_0_hat, S_0_hat_val

def antitonic_regression_pava(w_t, y_t, support=[-1/2, 1/2], **kwargs):
    inds_sorted = np.argsort(w_t)
    w_t = w_t[inds_sorted]
    y_t = y_t[inds_sorted]

    # check for duplicates in w_t
    unique_w_t, indices, counts = np.unique(w_t, return_index=True, return_counts=True)
    y_t_group = np.array([np.mean(y_t[w_t == u]) for u in unique_w_t])
    w_t = unique_w_t

    # apply PAVA
    iso_regressor = IsotonicRegression(increasing=False)
    iso_regressor.fit(unique_w_t, y_t_group, sample_weight=counts)

    S_0_hat = lambda x: iso_regressor.predict(np.atleast_1d(x))
    S_0_hat_val = iso_regressor.predict(np.sort(np.unique(w_t)))
    return S_0_hat, S_0_hat_val






# algo from Bracale et al. 2025

def dynamic_pricing_bracale(X, v_samples, alpha,
                            tau_1, n_episodes, p_min = 0., p_max = 1., support = np.array([-0.5, 0.5])):
    
    """
    Dynamic pricing algorithm from Bracale et al. 2025.
    Input:
    - X: input data for each time step
    - v_samples: samples of the valuation function
    - alpha: Hoelder continuity parameter
    - tau_1: parameter for lengths of the first episode
    - num_timest: number of time steps
    - p_min, p_max: minimum and maximum prices
    - support: support of the noise distribution
    """

    H = p_max - p_min
    nu_alpha = lambda x: 1 / (2 + x) if x < 0.5 else (2*x + 1)/(3*x + 1)

    ind_exploit = []
    p_opt = np.zeros(len(v_samples))

    for k in range(1, n_episodes+1, 1):
        print(f"Epoch {k}")
        tau_k = tau_1 * 2**(k-1) # length of episode
        tau_k_1 = tau_1 * 2**k 
        # a_k = np.ceil((H**(alpha/(2 + alpha))) * (tau_k**nu_alpha(alpha)) / 2) # length of exploration phase
        a_k = np.ceil((H**(alpha/(2 + alpha))) * (tau_k**nu_alpha(alpha))) # length of exploration phase
        I_k  = np.arange(tau_k - tau_1, tau_k - tau_1 + a_k, dtype=int) # exploration phase estimatin theta
        II_k = np.arange(tau_k - tau_1 + a_k, tau_k - tau_1 + 2 * a_k, dtype=int) # exploration phase S_0

        # exploration phase
        # estimating theta
        x_k = X[:, I_k] # covariates in the exploration phase for theta
        p_samples = np.random.uniform(low=p_min, high=p_max, size=len(I_k))
        y_t = np.array(v_samples[I_k] >= p_samples, dtype=int)
        model_theta_hat = LinearRegression(fit_intercept=False)
        model_theta_hat.fit(x_k.T, H*y_t)
        theta_hat = model_theta_hat.coef_ # estimated theta
        p_opt[I_k] = p_samples # add prices from exploration phase to p_opt

        # estimating S_0
        x_k = X[:, II_k]
        w_samples = np.random.uniform(low=support[0], high=support[1], size=len(II_k))
        p_t = w_samples + model_theta_hat.predict(x_k.T)
        # p_t = w_samples + model_cheating.predict(x_k.T)
        y_t = np.array(v_samples[II_k] >= p_t, dtype=int)
        p_opt[II_k] = p_t  # add prices from exploration phase to p_opt

        # S_0_hat, _ = antitonic_regression(w_samples, y_t, support=support)
        # S_0_hat, S_0_hat_val = antitonic_regression_sklearn(w_samples, y_t, support=support)
        S_0_hat, __ = antitonic_regression_pava(w_samples, y_t, support=support)

        # exploitation phase
        # indices exploitation phase
        E_k = np.arange(tau_k - tau_1 + 2 * a_k, tau_k_1 - tau_1 - 1, dtype=int)
        ind_exploit += list(E_k)
        # p_t = argmax_{p\in [p_min, p_max]} p S_0(p - theta_hat dot x_k)

        # find p_t
        # objective = lambda p, t: -p * S_0_hat(p - t)
        for i in E_k:
            theta_dot_x = model_theta_hat.predict(X[:, i].reshape(1, -1))
            # theta_dot_x = model_cheating.predict(X[:, i].reshape(1, -1))
            # result = minimize_scalar(objective, args=(theta_dot_x,), bounds=(p_min, p_max), method='bounded')
            objective = lambda p: -p * S_0_hat(p - theta_dot_x)
            result = minimize_scalar(objective, bounds=(p_min, p_max), method='bounded')
            p_opt[i] = result.x


    return p_opt, S_0_hat, model_theta_hat, ind_exploit
