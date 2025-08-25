import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from concurrent.futures import ProcessPoolExecutor



import utils as ut
import densities as dn
from bracale import dynamic_pricing_bracale

# save_plots = True
save_plots = False
fig_name = 'figures/bracale_regret_alpha_0.pdf'
verbose = True



# setup
d = 3
tau_1 = 100
support = np.array([-1/2, 1/2])
num_alph = 0
n_runs = 36
n_episodes = 8
# n_runs = 2
# n_episodes = 3
# max_workers = os.cpu_count()
max_workers = 36
n_timesteps = 2**(n_episodes) * tau_1 - tau_1

# theta_0
alpha_0 = 3
beta_0 = np.array([2/3, 2/3, 2/3])
theta_0 = np.concatenate((beta_0, [alpha_0]))

alphas_hoelder = np.array([1/3, 1/2, 3/4])


# save_dir = 'data/bracale_' + str(n_runs)+'_' + str(n_episodes) + '_' + str(num_alph) + '_/'
save_dir = 'data/bracale_' + str(n_runs)+'_' + str(n_episodes) + '_' + str(num_alph) + '_fan/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



bound = np.sqrt(2 / 3)

def single_run(i):
    np.random.seed(42 + i)
    # covariates
    X = np.random.uniform(low=-bound, high=bound, size=(d, n_timesteps))
    X = np.vstack((X, np.ones(n_timesteps)))  # add intercept

    # valuation function -- ground truth
    z_samples, F_0 = dn.sample_noise_fan(n_timesteps, m=2, support=support, verbose=verbose)
    # z_samples, F_0 = dn.sample_bracale_gaussian_laplace_cauchy(n_timesteps, density='cauchy', verbose=verbose)
    # z_samples, F_0 = dn.sample_bracale_hoelder(n_timesteps, alphas_hoelder[num_alph], verbose=verbose)
    v_samples = np.dot(X.T, theta_0) + z_samples

    p_star = ut.p_star_oracle(X, theta_0, F_0, p_range=[0, 5], verbose=verbose)

    print(f'Run {i+1}/{n_runs}')
    p_pred, model_S_0_hat, mod_theta_hat, inds_expl = dynamic_pricing_bracale(X, v_samples, alpha=alphas_hoelder[0],
                                                tau_1=tau_1, n_episodes=n_episodes,
                                                p_min=0., p_max=5., support=support)

    regret = ut.empirical_regret(p_star, p_pred, v_samples, verbose=verbose)
    return regret

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    regret_all = list(executor.map(single_run, range(n_runs)))

_, F_0 = dn.sample_bracale_hoelder(n_timesteps, alphas_hoelder[num_alph], verbose=verbose)

print('Finished all runs')

regret_all = np.array(regret_all)

mean_regret = np.mean(regret_all, axis=0)
std_regret = np.std(regret_all, axis=0)
confidence_level = 0.95
z_value = ut.find_quantile(F_0, confidence_level)
conf_int_margin = z_value * (std_regret / np.sqrt(n_runs))
conf_int_lower = mean_regret - conf_int_margin
conf_int_upper = mean_regret + conf_int_margin


# save regret_all, conf_int_margin, mean_regret, std_regret
np.savez(save_dir + "regret_results.npz",
         regret_all=regret_all,
         std_regret=std_regret,
         mean_regret=mean_regret,
         conf_int_margin=conf_int_margin)

print(f'Saved results to {save_dir}regret_results.npz')




# # plot regret
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(np.arange(n_timesteps, dtype=int), regret_all.T, label='regret')
# ax.fill_between(np.arange(n_timesteps, dtype=int), conf_int_lower, conf_int_upper, alpha=0.2, label=f'{confidence_level*100:.0f}% CI')
# # ax.plot(inds_expl, [0] * len(inds_expl), 'ro', markersize=1, label='exploration phase')
# ax.set_xlabel('log(t)')
# ax.set_ylabel('log(R(t))')
# ax.set_xscale('log', base=2)
# ax.set_yscale('log', base=2)
# ax.legend()
# plt.title('Log Regret over Log Time $\log(R(t))/ \log(t)$')
# plt.show()

    




# log_regret_all = np.log2(regret_all, where=regret_all > 0)
# log_tms = np.log2(np.arange(1, n_timesteps + 1, dtype=int))
#
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
#
#
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(log_tms, mean_regret_log, label='mean log regret')
# ax.fill_between(log_tms, conf_int_lower, conf_int_upper, alpha=0.2, label=f'{confidence_level*100:.0f}% CI')
# ax.plot(log_tms, log_tms * slope + fit[1], label=f'linear fit, slope={slope:.2f}', linestyle='--')
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
# ax.plot(x_indices, p_pred[:n_timesteps], 'o', markersize=1, label='p_pred')
# ax.plot(x_indices, p_star[:n_timesteps], 'o', markersize=1, label='p_star')
# # ax.plot(x_indices[condition], np.zeros_like(x_indices[condition]), 'o', markersize=1, label='$v_t > \hat p_t$')
# ax.plot(inds_expl, [0] * len(inds_expl), 'ro', markersize=1, label='exploration phase')
# ax.set_xlabel('t')
# ax.set_ylabel('v_samples, p_pred')
# ax.legend()
# plt.show()


if False:
    z_samples_sorted = np.sort(z_samples)
    v_samples_sorted = np.sort(v_samples)
    S_0_hat_samples = model_S_0_hat(z_samples_sorted.reshape(-1, 1))

    # plot F_0 
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    x_vals = np.linspace(-1/2, 1/2, n_timesteps)
    ax.plot(ut.ecdf(z_samples)[0], 1 - ut.ecdf(z_samples)[1], label='ECDF z_samples')
    ax.plot(z_samples_sorted, S_0_hat_samples, label='S_0_hat')
    ax.set_xlabel('x')
    ax.set_ylabel('F_0')
    ax.legend()
    plt.show()




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


# plot_log_regret_mod(log_regret_all, log_tms, regret_all, confidence_level=0.95, save_plots=save_plots, file_name=fig_name)
# plot_log_regret_mod(log_regret_all, log_tms, regret_all, confidence_level=0.95, save_plots=False, file_name=fig_name)
# plt.show()


