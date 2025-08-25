import numpy as np
from scipy.stats import truncnorm, cauchy, laplace
import matplotlib.pyplot as plt

import utils as ut


def rejection_sampling(f, max_f, n_samples, support=[-1/2, 1/2]):
    samples = []
    while len(samples) < n_samples:
        x = np.random.uniform(low=support[0], high=support[1], size=n_samples - len(samples))
        u = np.random.uniform(low=0, high=max_f, size=n_samples - len(samples))
        samples.extend(x[u <= f(x)])
    return np.array(samples)




# =============================================
# Densities Noise
# =============================================
# Bracale
# ---------------------------------------------
def sample_bracale_hoelder(n_samples, alpha, verbose=False):
    """
    Generate samples from the Bracale distribution with Hoelder exponent alpha < 1.
    """
    assert alpha < 1, "alpha must be less than 1"
    assert alpha > 0, "alpha must be greater than 0"

    F_0 = lambda x, alpha: np.clip(0.5 + 0.5**(1 - alpha) * np.sign(x) * np.abs(x)**alpha, 0, 1)
    F_inv = lambda x, alpha: np.sign(x - 0.5) * (2**(1 - alpha) * np.abs(x - 0.5))**(1 / alpha)
    z_samples = F_inv(np.random.uniform(0, 1, size=n_samples), alpha)  # sample from the distribution

    F_alpha = lambda x: F_0(x, alpha)

    ut.verbose_print(f"sampled from distribution with Hoelder exponent {alpha}", verbose)
    return z_samples, F_alpha

def sample_bracale_fan(n_samples, support=[-1/2, 1/2], verbose=False):

    f_0 = np.vectorize(lambda x: np.max([6*(0.25 - x**2), 0]))

    z_samples = rejection_sampling(f_0, f_0(0), n_samples, support=support)

    ut.verbose_print(f"sampled from distribution with f_0", verbose)
    return z_samples, f_0
    
def sample_bracale_gaussian_laplace_cauchy(n_samples, density='gaussian', support=[-1/2, 1/2], verbose=False):
    """
    Generate samples from the distributions used in Bracale et al. with truncated Gaussian, Laplace or Cauchy noise.
    """
    assert density in ['gaussian', 'laplace', 'cauchy'], "density must be one of 'gaussian', 'laplace' or 'cauchy'"

    if density == 'gaussian':
        a, b = support[0], support[1]
        z_samples = truncnorm.rvs(a=a, b=b, loc=0, scale=1, size=n_samples)
    elif density == 'laplace':
        z_samples = []
        while len(z_samples) < n_samples:
            x = laplace.rvs(loc=0, scale=0.2, size=n_samples - len(z_samples))
            z_samples.extend(x[(x >= support[0]) & (x <= support[1])])
        z_samples = np.array(z_samples)
    elif density == 'cauchy':
        z_samples = []
        while len(z_samples) < n_samples:
            x = cauchy.rvs(loc=0, scale=0.2, size=n_samples - len(z_samples))
            z_samples.extend(x[(x >= support[0]) & (x <= support[1])])
        z_samples = np.array(z_samples)


    if density == 'gaussian':
        F_0 = lambda x: truncnorm.cdf(x, a=a, b=b, loc=0, scale=1)
    elif density == 'laplace':
        F_0 = lambda x: laplace.cdf(x, loc=0, scale=0.2)
    elif density == 'cauchy':
        F_0 = lambda x: cauchy.cdf(x, loc=0, scale=0.2)

    ut.verbose_print(f"sampled from truncated {density} distribution on support {support}", verbose)
    return z_samples, F_0





# Fan
# ---------------------------------------------
def sample_noise_fan(n_samples, m=2, support=[-1/2, 1/2], verbose=False):
    """
    Generate samples from the distribution in Fan et al. (2022).
    """
    indicator = lambda x: (x >= support[0]) & (x <= support[1])
    indicator_end = lambda x: x > support[1]

    if m == 2:
        a_m = 6
        F_m = np.vectorize(lambda x: ((3*x)/2 - 2*x**3 + 1/2) * indicator(x) + indicator_end(x))
    elif m == 4:
        a_m = 30
        F_m = np.vectorize(lambda x: ((15*x)/8 - 5*x**3 + 6*x**5 + 1/2) * indicator(x) + indicator_end(x))
    else:
        a_m = 140
        F_m = np.vectorize(lambda x: ((35*x)/16 - (35*x**3)/4 + 21*x**5 - 20*x**7 + 1/2) * indicator(x) + indicator_end(x))


    f_m = np.vectorize(lambda x: a_m * (0.25 - x**2)**(m/2) * indicator(x))
    z_samples = rejection_sampling(f_m, f_m(0), n_samples, support=support)

    ut.verbose_print(f"sampled from distribution with f_m", verbose)
    return z_samples, F_m



# Non-Hoelder denisty
# ---------------------------------------------
def sample_noise_non_hoelder(n_samples, supp='small', verbose=False):
    if supp == 'small':
        support = [-0.5, 0.5]
    elif supp == 'large':
        support = [-1, 1]
    indicator = lambda x: (x > support[0]) & (x < support[1])

    if supp == 'small':
        # f = np.vectorize(lambda x: 30 if x == 0 else (np.where(x < 0, 1, 0) / (-1 + np.log(-2 * x))**2 + np.where(x >= 0, 1, 0) / (-1 + np.log(2 * x))**2) * (1 / (4 * x)) * indicator(x))
        def f(x):
            x = np.atleast_1d(x)
            result = np.zeros_like(x, dtype=float)
            mask_neg = x < 0
            mask_pos = x > 0
            result[mask_neg] = -1 / (np.log(-2 * x[mask_neg]) - 1)**2 * (1 / (4 * x[mask_neg])) * indicator(x[mask_neg])
            result[mask_pos] = 1 / (np.log(2 * x[mask_pos]) - 1)**2 * (1 / (4 * x[mask_pos])) * indicator(x[mask_pos])
            # result[x == 0] = 30
            return result

    elif supp == 'large':
        f = np.vectorize(lambda x: 30 if x == 0 else 1 / (2 * np.abs(x) * (1 - np.log(np.abs(x)))**2) * indicator(x))
    def F(x):
        if supp == 'small':
            x = 2*x
        result = np.zeros_like(x, dtype=float)
        mask_neg = x < 0
        mask_pos = x > 0
        result[mask_neg] = (1/4) * ((1 - np.sign(x[mask_neg])) / (np.log(-x[mask_neg]) - 1)) + 0.5
        result[mask_pos] = (1/4) * ((1 + np.sign(x[mask_pos])) / (1 - np.log(x[mask_pos]))) + 0.5
        result[x == 0] = 0.5
        result = np.clip(result, 0, 1)
        return result

    # z_samples = F_inv(np.random.uniform(0, 1, size=n_samples))

    max_val = f(0.00001)
    z_samples = rejection_sampling(f, max_val, n_samples, support=support)

    ut.verbose_print("sampled from non-Hoelder distribution", verbose)
    return z_samples, F








# =============================================
# Densities covariates X_t
# =============================================
# Bracale
# ---------------------------------------------
def sample_X_bracale(n_samples, d, bound=np.sqrt(2/3), verbose=False):
    """
    Generate uniform samples b/w [-bound, bound] in d dimensions.
    """
    assert d > 0, "d must be greater than 0"
    assert bound > 0, "bound must be greater than 0"

    X = np.random.uniform(low=-bound, high=bound, size=(d, n_samples))
    X = np.vstack((X, np.ones(n_samples)))  # add intercept

    ut.verbose_print(f"sampled from uniform distribution on [-{bound}, {bound}] in {d} dimensions", verbose)
    return X




# Fan
# ---------------------------------------------
def sample_X_fan(n_samples, d, m=2, mode='iid_indep', bound=np.sqrt(2/3), verbose=False):
    """
    Generate uniform samples b/w [-bound, bound] in d dimensions.
    """
    assert d > 0, "d must be greater than 0"
    assert bound > 0, "bound must be greater than 0"
    assert m in [2, 4, 6], "m must be one of 2, 4, 6"

    indicator = lambda x: np.abs(x) <= bound

    if mode == 'iid_indep':
        if m == 2:
            a_m = 945 / (256*np.sqrt(2/3))
        if m == 4:
            a_m = 1 / 0.07943918896
        if m == 6:
            a_m = 1 / 0.03041774073
            

        f_m = np.vectorize(lambda x: a_m * (2/3 - x**2)**(m + 1) * indicator(x))
        x_samples = rejection_sampling(f_m, f_m(0), n_samples * d, support=[-bound, bound])
        X = x_samples.reshape((d, n_samples))

    return X


if __name__ == "__main__":
    zs, _ = sample_noise_non_hoelder(60000, supp='small', verbose=True)
    plt.hist(zs, bins=50, range=(-0.5, 0.5))
    plt.show()

