import numpy as np
import matplotlib.pyplot as plt
import os

from concurrent.futures import ProcessPoolExecutor

import utils as ut
import densities as dn
from new_algo_1 import single_index_profile
from plotting import plot_log_regret
from verifying_estimator import get_noise


# save_plots = True
save_plots = False
fig_name = 'figures/new_regret.pdf'
verbose = True

# density = 'bracale'
# density = 'fan'
density = 'not_hoelder'
# density = 'uniform'


# setup
d = 3
l_0 = 100
support = np.array([-1/2, 1/2])
n_episodes = 8
n_runs = 36
# n_runs = 12
# n_runs = 1
# n_episodes = 3
# n_timesteps = 2**(n_episodes) * tau_1 - tau_1
n_timesteps = l_0 * (2**(n_episodes + 1) - 2)

# max_runners = os.cpu_count()
max_runners = n_runs

# save_dir = 'data/new_' + str(n_runs) + '_' + str(n_episodes) + '_grid_2_' + density + '/'
save_dir = 'data/new_' + str(n_runs) + '_' + str(n_episodes) + '_grid_2_' + density + '_small/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# theta_0
alpha_0 = 3
beta_0 = np.array([2/3, 2/3, 2/3])
theta_0 = np.concatenate((beta_0, [alpha_0]))

bound = np.sqrt(2 / 3)

def single_run(i):
    np.random.seed(42 + i)
    # covariates
    X = np.random.uniform(low=-bound, high=bound, size=(d, n_timesteps))
    # X = dn.sample_X_fan(n_timesteps, d=d, verbose=verbose)
    X = np.vstack((X, np.ones(n_timesteps)))  # add intercept

    # valuation function -- ground truth
    if density == 'bracale':
        z_samples, F_0 = get_noise(n_timesteps, params=[1/3], density='bracale_hoelder')
    elif density == 'fan':
        z_samples, F_0 = get_noise(n_timesteps, params=[2, [-1/2, 1/2]], density='fan')
    elif density == 'not_hoelder':
        z_samples, F_0 = get_noise(n_timesteps, params=[1/3], density='not_hoelder')
    elif density == 'uniform':
        z_samples, F_0 = get_noise(n_timesteps, params=[-1/2, 1/2], density='uniform')

    v_samples = np.dot(X.T, theta_0) + z_samples

    # optimal price
    p_star = ut.p_star_oracle(X, theta_0, F_0, p_range=[0, 5], verbose=verbose)

    print(f'Run {i+1}/{n_runs}')
    p_pred = single_index_profile(X, v_samples, l_0=l_0, num_episodes=n_episodes)
    # psi_vals.append(psi_val)
    # theta_hats.append(theta_hat)
    regret = ut.empirical_regret(p_star, p_pred, v_samples, verbose=verbose)
    # return (regret, p_pred, X)
    return regret

with ProcessPoolExecutor(max_workers=max_runners) as executor:
    regret_all = list(executor.map(single_run, range(n_runs)))
    # results = list(executor.map(single_run, range(n_runs)))

if False:
    regret_all = []
    p_pred_all = []
    X_all = []
    for i in range(len(results)):
        regret, p_pred, X = results[i]
        regret_all.append(regret)
        p_pred_all.append(p_pred)
        X_all.append(X)

    regret_all = np.array(regret_all)
    p_pred_all = np.array(p_pred_all)
    X_all = np.array(X_all)

    counts_regret = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[1], axis=0, arr=regret_all)
    print("counts regret ", counts_regret)
    counts_p_pred = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[1], axis=0, arr=p_pred_all)
    print("counts p_pred ", counts_p_pred)
    counts_X = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[1], axis=0, arr=X_all)
# counts_X = np.sum(counts_X)
    print("counts X ", counts_X)

    

_, F_0 = dn.sample_noise_fan(n_timesteps, m=2, support=support, verbose=verbose)
print('Finished all runs')

# psi_vals = np.array(psi_vals[0])
regret_all = np.array(regret_all)

std_regret = np.std(regret_all, axis=0)
mean_regret = np.mean(regret_all, axis=0)
confidence_level = 0.95
z_value = ut.find_quantile(F_0, confidence_level)
conf_int_margin = z_value * (std_regret / np.sqrt(n_runs))
conf_int_lower = mean_regret - conf_int_margin
conf_int_upper = mean_regret + conf_int_margin


# Save results
# if False:
if True:
    np.savez(save_dir + 'regret_results.npz',
             regret_all=regret_all,
             std_regret=std_regret,
             mean_regret=mean_regret,
             conf_int_margin=conf_int_margin)

print(f'Saved results to {save_dir}regret_results.npz')





log_regret_all = np.log2(regret_all, where=regret_all > 0)
log_tms = np.log2(np.arange(1, n_timesteps + 1, dtype=int))

plot_log_regret(log_tms, log_regret_all)
plt.show()

# mean_regret_log = np.mean(log_regret_all, axis=0)
# mean_regret_log_2 = np.log2(mean_regret, where=mean_regret > 0)
#
# # fit = LinearRegression().fit(log_tms.reshape(-1, 1), mean_regret_log)
# fit = np.polyfit(log_tms, mean_regret_log, 1)
# slope = fit[0]
#
#
#
# std_regret_log = np.std(log_regret_all, axis=0)
# confidence_level = 0.95
# z_value = ut.find_quantile(F_0, confidence_level)
# conf_int_margin = z_value * (std_regret_log / np.sqrt(n_runs))
# conf_int_lower = mean_regret_log - conf_int_margin
# conf_int_upper = mean_regret_log + conf_int_margin
#
# episode_starts = [1]
# for k in range(1, n_episodes + 1):
#     episode_starts.append(episode_starts[-1] + l_0 * 2**k)
#
#
#
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(log_tms, mean_regret_log, label='mean log regret')
# # ax.plot(log_tms, log_regret_all.T, 'o', markersize=1, alpha=0.1, label='log regret samples')
# ax.plot(log_tms, log_regret_all.T, label='log regret samples')
# ax.fill_between(log_tms, conf_int_lower, conf_int_upper, alpha=0.2, label=f'{confidence_level*100:.0f}% CI')
# ax.plot(log_tms, log_tms * slope + fit[1], label=f'linear fit, slope={slope:.2f}', linestyle='--')
# ax.plot(np.log2(episode_starts), np.zeros_like(episode_starts), 'ko', markersize=3, label='Episode starts')
# ax.set_xlabel('log(t)')
# ax.set_ylabel('log(R(t))')
# ax.set_title('Log Regret over Log Time $\log(R(t))/ \log(t)$ with Linear Fit')
# ax.legend()
# if save_plots:
#     # fig.savefig('figures/bracale_fan_regret_m_2.pdf', format='pdf')
#     fig.savefig(fig_name, format='pdf')
# plt.show()
#
#
# x_indices = np.arange(n_timesteps, dtype=int)
# condition = v_samples[:n_timesteps] > p_pred[:n_timesteps]
#
#
# # plot v_t and estimated p_t
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(x_indices, v_samples[:n_timesteps], 'o', markersize=1, label='v_samples')
# ax.plot(x_indices, p_star[:n_timesteps], 'o', markersize=1, label='p_star')
# ax.plot(x_indices, p_pred[:n_timesteps], 'o', markersize=1, label='p_pred')
# # ax.plot(x_indices[condition], np.zeros_like(x_indices[condition]), 'o', markersize=1, label='$v_t > \hat p_t$')
# ax.plot(episode_starts, [0]*len(episode_starts), 'ko', markersize=3, label='Episode starts')
# ax.set_xlabel('t')
# ax.set_ylabel('v_samples, p_pred')
# ax.legend()
# plt.show()
