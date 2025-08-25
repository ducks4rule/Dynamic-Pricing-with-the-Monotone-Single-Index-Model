import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

import utils as ut
import densities as dn


def compute_confidence_intervals_bootstr(regret, conf_level=0.95, n_resamples=1000):
    n_samples, n_pts = regret.shape
    resample_idx = np.random.randint(0, n_samples, size=(n_resamples, n_samples))
    # bootstraped_means = np.mean(regret[resample_idx, :, np.newaxis], axis=1)
    bootstrapped_means = np.mean(regret[resample_idx], axis=1)

    conf_lower = np.percentile(bootstrapped_means, conf_level / 2 * 100, axis=0)
    conf_upper = np.percentile(bootstrapped_means, (1 - conf_level / 2) * 100, axis=0)
    mean_regret = np.mean(regret, axis=0)

    return mean_regret, conf_lower, conf_upper

def compute_confidence_intervals_normal(regret, conf_level=0.95):
    mean_regret = np.mean(regret, axis=0)
    std_err = stats.sem(regret, axis=0) # standard error of the mean
    z_critical = stats.norm.ppf((1 + conf_level) / 2)  # z-score for the confidence level
    margin_of_error = z_critical * std_err

    conf_lower = mean_regret - margin_of_error
    conf_upper = mean_regret + margin_of_error
    return mean_regret, conf_lower, conf_upper

def plot_log_regret_mod(log_regret, log_tms, regret_all,
                        confidence_level=0.95, save_plots=False,
                        fontsize=16,
                        show_legend=True,
                        show_title=False,
                        file_name='figures/fan_fan_regret_m_2.pdf'):
    sel_tms = np.array([1500, 2000, 3100, 4000, 5000, 6300])
    mod_tms = np.log2(np.array(sel_tms)) - np.log2(1500)
    mod_reg_inds = np.abs(log_tms[:, None] - np.log2(sel_tms)).argmin(axis=0)
    log_reg_mod = log_regret[:, mod_reg_inds] - 2 * np.log2(log_tms[mod_reg_inds]).reshape(1, -1) \
        - (np.log2(regret_all[:, 1500 - 1]).reshape(-1, 1) - 2 * np.log2(np.log2(1500)))
    fit_mod = np.polyfit(mod_tms, np.mean(log_reg_mod, axis=0), 1)
    slope_mod = fit_mod[0]

    # Compute confidence intervals
    mean_log_reg_mod = np.mean(log_reg_mod, axis=0)
    std_log_reg_mod = np.std(log_reg_mod, axis=0, ddof=1)
    n = log_reg_mod.shape[0]
    # _, conf_int_lower_mod, conf_int_upper_mod = compute_confidence_intervals_bootstr(log_reg_mod, confidence_level, n_resamples=100)
    _, conf_int_lower_mod, conf_int_upper_mod = compute_confidence_intervals_normal(log_reg_mod, confidence_level)
    

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(mod_tms, log_reg_mod.T, label='log regret mod', linestyle='-')
    ax.fill_between(mod_tms, conf_int_lower_mod, conf_int_upper_mod, alpha=0.2, label=f'{confidence_level*100:.0f}% CI mod')
    ax.plot(mod_tms, mod_tms * slope_mod + fit_mod[1], label=f'linear fit mod, slope={slope_mod:.2f}', linestyle='--')
    ax.set_xlabel('log(T) - log(1500))', fontsize=fontsize)
    ax.set_ylabel('log(R(t)) - 2 * log(log(T)) - (log(R(1500)) - 2 * log(log(1500)))', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    if show_title:
        ax.set_title('Log Regret as in Fan et al.', fontsize=fontsize)
    if show_legend:
        ax.legend(fontsize=fontsize)
    if save_plots:
        fig.savefig(file_name, format='pdf')
        ut.verbose_print(f"Plot saved to {file_name}")
    return fig


def plot_log_regret(log_regret: np.ndarray,  # shape (n_runs, n_timesteps)
                    log_tms: np.ndarray,
                    save_plot=False, confidence_level=0.95,
                    fontsize=16,
                    show_legend=True,
                    show_title=False,
                    file_name='figures/fan_fan_regret_m_2.pdf',
                    plot_mode='all', plot_confidence_intervals=True):
    assert plot_mode in ['all', 'mean', 'median', 'mod'], "plot_mode must be 'all', 'mean', or 'median'"

    if plot_mode == 'mod':
        return plot_log_regret_mod(log_regret, log_tms, regret_all,
                                   confidence_level=confidence_level,
                                   fontsize=fontsize, show_title=show_title,
                                   save_plots=save_plot, show_legend=show_legend,
                                   file_name=file_name)

    # mean_log_regret, conf_int_lower, conf_int_upper = compute_confidence_intervals_bootstr(log_regret, confidence_level, n_resamples=10)
    mean_log_regret, conf_int_lower, conf_int_upper = compute_confidence_intervals_normal(log_regret, confidence_level)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    if plot_mode == 'mean':
        ax.plot(log_tms, mean_log_regret, label='mean log regret', color='blue')
    elif plot_mode == 'median':
        mean_log_regret = np.median(log_regret, axis=0)
        ax.plot(log_tms, mean_log_regret, label='median log regret', color='orange')
    else:
        ax.plot(log_tms, log_regret.T, label='log regret')
    if plot_confidence_intervals:
        ax.fill_between(log_tms, conf_int_lower, conf_int_upper,
                        alpha=0.2, label=f'{confidence_level*100:.0f}% CI')
    slope, intercept = np.polyfit(log_tms, mean_log_regret, 1)
    ax.plot(log_tms, log_tms * slope + intercept, label=f'linear fit, slope={slope:.2f}', linestyle='--')
    ax.set_xlabel('log(t)', fontsize=fontsize)
    ax.set_ylabel('log(R(t))', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    if show_title:
        ax.set_title('Log Regret over Log Time $\log(R(t))/ \log(t)$ with Linear Fit', fontsize=fontsize)
    if show_legend:
        ax.legend(fontsize=fontsize)
    if save_plot:
        fig.savefig(file_name, format='pdf')
        ut.verbose_print(f"Plot saved to {file_name}")
    return fig


def plot_regret_comparison(load_folders,
                            confidence_level=0.95,
                            save_plot=False,
                            fontsize=16,
                            show_legend=True,
                            show_title=False,
                            file_name='figures/verification/regret_comparison.pdf'):
    regrets = []
    mean_regrets = []
    log_regrets = []
    mean_log_regrets = []
    conf_int_bds = []
    for folder in load_folders:
        data = np.load(folder + 'regret_results.npz')
        regrets.append(data['regret_all'])
        mean_regrets.append(data['mean_regret'])
        log_regret = np.log2(data['regret_all'], where=data['regret_all'] > 0)
        log_regrets.append(log_regret)
        mean_log_regret, conf_int_lower, conf_int_upper = compute_confidence_intervals_normal(log_regret, confidence_level)
        conf_int_bds.append((conf_int_lower, conf_int_upper))
        mean_log_regrets.append(mean_log_regret)

    Ts = np.array([500, 1000, 2000, 4000, 8000])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    for i, (regret, mean_regret, log_regret, conf_int_bd) in enumerate(zip(regrets, mean_regrets, log_regrets, conf_int_bds)):
        if i == 0:
            label = 'Fan et al.'
        elif i == 1:
            label = 'Bracale et al.'
        else:
            label = 'MSIM policy'
        
        ax.plot(Ts, mean_regret[Ts], 'o-', label=label)
        ax.fill_between(Ts, conf_int_bd[0][Ts], conf_int_bd[1][Ts], alpha=0.2)
        ax.set_xlabel('Time Horizon T', fontsize=fontsize)
        ax.set_ylabel('log(R(T))', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        if show_title:
            ax.set_title('Log Regret Comparison', fontsize=fontsize)
        if show_legend:
            ax.legend(fontsize=fontsize)
    if save_plot:
        fig.savefig(file_name, format='pdf')
        ut.verbose_print(f"Plot saved to {file_name}")
    return fig
        

if __name__ == "__main__":
    np.random.seed(42)
    # ==============================
    # Algorithm and data parameters
    # ==============================
    
    # which_algo = 'fan'
    # which_algo = 'bracale'
    which_algo = 'new'

    # Set parameters for data import
    # folder = 'data/fan_36_8_/'
    # folder = 'data/bracale_36_8_0_/'
    # folder = 'data/bracale_36_8_0_fan/'
    # folder = 'data/new_36_8_grid_2/'
    density = None
    # density = 'bracale'
    # density = 'uniform'
    # density = 'fan'
    density = 'not_hoelder'
    # folder = 'data/new_36_8_grid_2_' + density + '/'
    folder = 'data/new_36_8_grid_2_' + density + '_small/'
    file_name = 'regret_results.npz'
    data = np.load(folder + file_name)

    # ==============================
    # Plotting and saving parameters
    # ==============================
    
    # save_plots = False
    save_plots = True
    # plot_verification = False
    plot_verification = True
    plot_mode = 'mean'
    # plot_mode = 'all'
    # plot_mode = 'mod'
    if plot_mode in ['all', 'mod']:
        show_legend = False
    else:
        show_legend = True
    # show_title = True
    show_title = False

    fontsize = 16
        
    if density is not None:
        save_dir = 'figures/' + which_algo + '_' + density + '/'
    else:
        save_dir = 'figures/' + which_algo + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name_save = save_dir + which_algo + '_36_8_' + plot_mode + '.pdf'

    # Extract data from the loaded file
    regret_all = data['regret_all']
    std_regret = data['std_regret']
    mean_regret = data['mean_regret']
    conf_int_margin = data['conf_int_margin']

    # Modify regret (take log2)
    log_regret_all = np.log2(regret_all, where=regret_all > 0)
    log_tms = np.log2(np.arange(1, regret_all.shape[1] + 1, dtype=int))

    # Gemerate F_0 distribution for confidence intervals
    # _, F_0 = dn.sample_bracale_fan(len(log_tms), verbose=True)
    # _, F_0 = dn.sample_noise_fan(len(log_tms), m=2, support=[0, 5], verbose=True)
    # _, F_0 = dn.sample_bracale_hoelder(len(log_tms), alpha=1./3, verbose=True)

    if not plot_verification:
        plot_log_regret(log_regret_all, log_tms,
                        save_plot=save_plots, confidence_level=0.95,
                        show_legend=show_legend,
                        file_name=file_name_save,
                        plot_mode=plot_mode,
                        fontsize=fontsize,
                        show_title=show_title,
                        plot_confidence_intervals=True)
    if plot_verification:
        plot_regret_comparison(['data/fan_36_8_unif/',
                                'data/bracale_36_8_0_fan/',
                                'data/new_36_8_grid_2_fan/'],
                                confidence_level=0.95,
                                save_plot=save_plots,
                                fontsize=fontsize,
                                show_legend=show_legend,
                                show_title=show_title)
                            
    plt.show()
