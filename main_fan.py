import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
import os
from concurrent.futures import ProcessPoolExecutor

import utils as ut
import densities as dn
from fan import dynamic_pricing_fan
# from fan_2 import dynamic_pricing_fan

save_plots = True
# save_plots = False
file_name = 'figures/fan_fan_regret_m_2.pdf'
verbose = True



# setup
d = 3
m = 2
tau_1 = 100
n_runs = 36
n_episodes = 8
# n_runs = 3
# n_episodes = 5
n_timesteps = 2**(n_episodes - 1) * tau_1
support = np.array([-1/2, 1/2])
# max_workers = os.cpu_count()
max_workers = 36


# save_dir = 'data/fan_' + str(n_runs)+'_' + str(n_episodes) + '_/'
save_dir = 'data/fan_' + str(n_runs)+'_' + str(n_episodes) + '_unif/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# theta_0
alpha_0 = 3
beta_0 = np.array([2/3, 2/3, 2/3])
theta_0 = np.concatenate((beta_0, [alpha_0]))

alphas_hoelder = np.array([1/3, 1/2, 3/4])

# covariates
bound = np.sqrt(2 / 3)
# X = np.random.uniform(low=-bound, high=bound, size=(d, n_timesteps))

def single_run(i):
    np.random.seed(42 + i)
    
    # X = dn.sample_X_fan(n_timesteps, d, m=m, mode='iid_indep', bound=bound, verbose=verbose)
    # X = np.vstack((X, np.ones(n_timesteps)))  # add intercept
    X = dn.sample_X_bracale(n_timesteps, d, bound=bound, verbose=verbose)

    # valuation function -- ground truth
    z_samples, F_0 = dn.sample_noise_fan(n_timesteps, m=m, support=support, verbose=verbose)
    # z_samples, F_0 = dn.sample_bracale_gaussian_laplace_cauchy(n_timesteps, density='cauchy', verbose=verbose)
    v_samples = np.dot(X.T, theta_0) + z_samples
    # v_t = lambda x: np.dot(x, theta_0) + z_samples

    p_star = ut.p_star_oracle(X, theta_0, F_0, p_range=[0, 5], verbose=verbose)

    print(f'Run {i+1}/{n_runs}')
    p_pred, inds_expl = dynamic_pricing_fan(X, v_samples, m=m, l_0=tau_1, num_episodes=n_episodes)

    regret = ut.empirical_regret(p_star, p_pred, v_samples, verbose=verbose)
    return regret

# Run the dynamic pricing algorithm in parallel
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    regret_all = list(executor.map(single_run, range(n_runs)))


print('Finished all runs')

# for i in range(n_runs):
#     print(f'Run {i+1}/{n_runs}')
#     p_pred, inds_expl = dynamic_pricing_fan(X, v_samples, m=m, l_0=tau_1, num_episodes=n_episodes)
#
#     regret = ut.empirical_regret(p_star, p_pred, v_samples, verbose=verbose)
#     regret_all.append(regret)
_, F_0 = dn.sample_noise_fan(n_timesteps, m=m, support=support, verbose=verbose)


regret_all = np.array(regret_all)

std_regret = np.std(regret_all, axis=0)
mean_regret = np.mean(regret_all, axis=0)
confidence_level = 0.95
z_value = ut.find_quantile(F_0, confidence_level)
conf_int_margin = z_value * (std_regret / np.sqrt(n_runs))
conf_int_lower = mean_regret - conf_int_margin
conf_int_upper = mean_regret + conf_int_margin


# Save results
np.savez(save_dir + "regret_results.npz",
         regret_all=regret_all,
         std_regret=std_regret,
         mean_regret=mean_regret,
         conf_int_margin=conf_int_margin)

print('Saved results to', save_dir + "regret_results.npz")




# log_regret_all = np.log2(regret_all, where=regret_all > 0)
# log_tms = np.log2(np.arange(1, n_timesteps + 1, dtype=int))
# slope, intercept = np.polyfit(log_tms, np.mean(log_regret_all, axis=0), 1)
# mean_regret_log = np.mean(log_regret_all, axis=0)
# conf_int_lower = mean_regret_log - conf_int_margin
# conf_int_upper = mean_regret_log + conf_int_margin
#
#
#
# std_regret_log = np.std(log_regret_all, axis=0)
# confidence_level = 0.95
# z_value = ut.find_quantile(F_0, confidence_level)
# conf_int_margin = z_value * (std_regret_log / np.sqrt(n_runs))
#
#
#
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(log_tms, mean_regret_log, label='mean log regret')
# ax.plot(log_tms, log_regret_all.T, label='log regret')
# ax.fill_between(log_tms, conf_int_lower, conf_int_upper, alpha=0.2, label=f'{confidence_level*100:.0f}% CI')
# ax.plot(log_tms, log_tms * slope + intercept, label=f'linear fit, slope={slope:.2f}', linestyle='--')
# ax.set_xlabel('log(t)')
# ax.set_ylabel('log(R(t))')
# ax.set_title('Log Regret over Log Time $\log(R(t))/ \log(t)$ with Linear Fit')
# ax.legend()
# plt.show()
#
#
# def plot_log_regret_mod(log_regret_all, log_tms, regret_all, confidence_level=0.95, save_plots=False, file_name='figures/fan_fan_regret_m_2.pdf'):
#     sel_tms = np.array([1500, 2000, 3100, 4000, 5000, 6300])
#     mod_tms = np.log2(np.array(sel_tms)) - np.log2(1500)
#     mod_reg_inds = np.abs(log_tms[:, None] - np.log2(sel_tms)).argmin(axis=0)
#     log_reg_mod = log_regret_all[:, mod_reg_inds] - 2 * np.log2(log_tms[mod_reg_inds]).reshape(1, -1) \
#         - (np.log2(regret_all[:, 1500 - 1]).reshape(-1, 1) - 2 * np.log2(np.log2(1500)))
#     fit_mod = np.polyfit(mod_tms, np.mean(log_reg_mod, axis=0), 1)
#     slope_mod = fit_mod[0]
#
#     # Compute confidence intervals
#     mean_log_reg_mod = np.mean(log_reg_mod, axis=0)
#     std_log_reg_mod = np.std(log_reg_mod, axis=0, ddof=1)
#     n = log_reg_mod.shape[0]
#     h = ut.find_quantile(F_0, confidence_level) * (std_log_reg_mod / np.sqrt(n))
#     conf_int_lower_mod = mean_log_reg_mod - h
#     conf_int_upper_mod = mean_log_reg_mod + h
#
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(1, 1, 1)
#     ax.plot(mod_tms, log_reg_mod.T, label='log regret mod', linestyle='-')
#     ax.fill_between(mod_tms, conf_int_lower_mod, conf_int_upper_mod, alpha=0.2, label=f'{confidence_level*100:.0f}% CI mod')
#     ax.plot(mod_tms, mod_tms * slope_mod + fit_mod[1], label=f'linear fit mod, slope={slope_mod:.2f}', linestyle='--')
#     ax.set_xlabel('log(t)')
#     ax.set_ylabel('log(R(t))')
#     ax.set_title('Log Regret over Log Time $\\log(R(t))/ \\log(t)$ with Linear Fit')
#     ax.legend()
#     if save_plots:
#         fig.savefig(file_name, format='pdf')
#     return fig

# plot_log_regret_mod(log_regret_all, log_tms, regret_all, confidence_level=0.95, save_plots=save_plots, file_name=file_name)
# plt.show()


print('Finished plotting')










# # plot v_t and estimated p_t
# actual_ts = n_timesteps
# x_indices = np.arange(actual_ts, dtype=int)
# condition = v_samples[:actual_ts] > p_pred[:actual_ts]
#
#
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(x_indices, v_samples[:actual_ts], 'o', markersize=1, label='v_samples')
# ax.plot(x_indices, p_pred[:actual_ts], 'o', markersize=1, label='p_pred')
# ax.plot(x_indices, p_star[:actual_ts], 'o', markersize=1, label='p_star')
# ax.plot(x_indices[condition], np.zeros_like(x_indices[condition]), 'o', markersize=1, label='$v_t > \hat p_t$')
# # ax.plot(inds_expl, [0] * len(inds_expl), 'ro', markersize=1, label='exploration phase')
# ax.set_xlabel('t')
# ax.set_ylabel('v_samples, p_pred')
# ax.legend()
# plt.show()
#
#
# if True:
#     model_S_0_hat = np.vectorize(lambda x: 1 - F_0(x))
#     z_samples_sorted = np.sort(z_samples)
#     v_samples_sorted = np.sort(v_samples)
#     S_0_hat_samples = model_S_0_hat(z_samples_sorted.reshape(-1, 1))
#
#     # plot F_0 
#     fig = plt.figure(figsize=(10, 5))
#     ax = fig.add_subplot(1, 1, 1)
#     x_vals = np.linspace(-1/2, 1/2, n_timesteps)
#     ax.plot(ut.ecdf(z_samples)[0], 1 - ut.ecdf(z_samples)[1], label='ECDF z_samples')
#     ax.plot(z_samples_sorted, S_0_hat_samples, label='S_0_hat')
#     ax.set_xlabel('x')
#     ax.set_ylabel('F_0')
#     ax.legend()
#     plt.show()
