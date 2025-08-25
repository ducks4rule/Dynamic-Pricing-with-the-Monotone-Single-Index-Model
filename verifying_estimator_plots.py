import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import json
import pickle
import os
from scipy.stats import linregress

from verifying_estimator import get_noise

what_to_plot = 'theta_235' 
# what_to_plot = 'F_hat_3'

save_plots = True
# save_plots = False

# load json data
density = 'uniform'
# density = 'bracale_gaussian'
# density = 'bracale_cauchy'
# density = 'bracale_laplace'
# density = 'bracale_hoelder_0.3333333333333333'
# density = 'bracale_hoelder_0.5'
# density = 'bracale_hoelder_0.75'
# density = 'fan_2'
# density = 'fan_4'
# density = 'fan_6'
density = 'not hoelder'
add_small = '_small'
# add_small = ''
load_dir = 'data/norms_estimator/'
if what_to_plot == 'theta_235':
    load_file = 'theta_F_hat_norms_' + density + add_small + '.json'
elif what_to_plot == 'F_hat_3':
    load_file = 'F_hat_3_' + density + add_small + '.pkl'

if density == 'bracale_hoelder_0.3333333333333333':
    density = 'bracale_hoelder_0.33'
if density == 'not hoelder':
    density = 'not_hoelder'

save_dir = 'figures/verification/'
if what_to_plot == 'theta_235':
    save_file = 'theta_F_hat_norms_' + density + '.pdf'
elif what_to_plot == 'F_hat_3':
    save_file = 'F_hat_3_' + density + '.pdf'


if not os.path.exists(save_dir):
    os.makedirs(save_dir)



n_list = np.logspace(np.log10(100), np.log10(60000), num=30, dtype=int).tolist()

if what_to_plot == 'theta_235':
    fontsize = 18

    with open(load_dir + load_file, 'r') as f:
        data = json.load(f)
    # Extract the data
    if density == 'uniform' and ('small' not in add_small):
        theta_2 = data['2_theta']
        theta_3 = data['3_theta']
        theta_5 = data['5_theta']
    else:
        theta_2 = data['theta_norms2']
        theta_3 = data['theta_norms3']
        theta_5 = data['theta_norms5']



    # impirical convergence rate
    log_n = np.log(n_list)
    log_theta_2 = np.log(theta_2)
    log_theta_3 = np.log(theta_3)
    log_theta_5 = np.log(theta_5)

    slope_2, intercept_2,*_ = linregress(log_n, log_theta_2)
    slope_3, intercept_3,*_ = linregress(log_n, log_theta_3)
    slope_5, intercept_5,*_ = linregress(log_n, log_theta_5)
    empirical_line_2 = np.exp(intercept_2) * np.array(n_list)**slope_2


    # theoretical convergence rate line
    ref_line_2 = theta_2[0] * (np.array(n_list) / n_list[0])**(-1/3)
    
    # plot all thetas vs log x and F_hat vs log x
    color_list = ['#3566b6', '#3ca374', '#e17c05']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_list, theta_2, label=r"$d = 2$", marker='o', color=color_list[0])
    ax.plot(n_list, theta_3, label=r"$d = 3$", marker='o', color=color_list[1])
    ax.plot(n_list, theta_5, label=r"$d = 5$", marker='o', color=color_list[2])
    ax.set_xscale('log')
    ax.set_xlabel(r"Number of samples ($n$), from $100$ to $60,000$", fontsize=fontsize)
    ax.set_ylabel(r"$\|\theta - \hat\theta\|_2$", fontsize=fontsize)
    # ax.set_title(r"Convergence of $\|\theta - \hat\theta\|_2$ for $d \in \{2, 3, 5\}$")
    ax.set_xticks(n_list[::3] + [n_list[-1]])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    # ax.xaxis.set_major_locator(LogLocator(base=2.0, numticks=10))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    # -- convergence rate line
    ax.plot(n_list, ref_line_2, label=r"Theoretical convergence rate $n^{-1/3}$", linestyle="--", color=color_list[0])
    ax.plot(n_list, empirical_line_2, label=fr"Empirical fit for $d=2$: $n^{{{slope_2:.2f}}}$", linestyle=":", color=color_list[0])
    ax.legend()
    ax.set_xlim(min(n_list) / 1.1, max(n_list) * 1.1)
    ax.figure.canvas.draw()
    x_min, x_max = ax.get_xlim()
    ax.hlines(y=0, xmin=x_min, xmax=x_max, colors='black', linestyles='dashed')
    ax.legend(fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    if save_plots:
        plt.savefig(os.path.join(save_dir, save_file), bbox_inches='tight')
        print(f"Saved figure to {os.path.join(save_dir, save_file)}")
    plt.show()

    # save the emprical convergence rates to a txt file
    if save_plots:
        empirical_rates = {
            'slope_2': slope_2,
            'slope_3': slope_3,
            'slope_5': slope_5,
        }
        empirical_rates_file = os.path.join(save_dir, 'empirical_convergence_rates_' + density + '.txt')
        with open(empirical_rates_file, 'w') as f:
            for key, value in empirical_rates.items():
                f.write(f"{key}: {value}\n")
        print(f"Saved empirical convergence rates to {empirical_rates_file}")


if what_to_plot == 'F_hat_3':
    fontsize = 18

    if density in ['uniform', 'bracale_gaussian', 'bracale_laplace', 'bracale_cauchy', 'not_hoelder']:
        params_noise = [-1/2, 1/2]
        # x_lims = [-1.2, 1.2]
        # x_lims = [-0.8, 0.8]
        x_lims = [-0.6, 0.6]
    if density in ['bracale_gaussian', 'bracale_laplace', 'bracale_cauchy']:
        params_noise = [0, 1]
    elif 'bracale_hoelder' in density:
        density = 'bracale_hoelder'
        params_noise = [1/2]
        if '0.33' in density:
            params_noise = [1/3]
        elif '0.75' in density:
            params_noise = [3/4]
        x_lims = [-0.6, 0.6]
    elif 'fan' in density:
        density = 'fan'
        params_noise = [6, [-1/2, 1/2]] # [2, [-1/2, 1/2]], [4, [-1/2, 1/2]], [6, [-1/2, 1/2]]
        x_lims = [-0.6, 0.6]

    with open(os.path.join(load_dir, load_file), 'rb') as f:
        data = pickle.load(f)

    # if density == 'uniform':
    if density == 'uniform' and ('small' not in add_small):
        data = data['3_theta']
        x_lims = [-0.2, 1.2]
    # x_lims = [-1.2, 1.2]
        
    # data = data['3_theta']

    F_0 = get_noise(n_samples=1000, params=params_noise, density=density)[1]
    n_list = np.logspace(np.log10(100), np.log10(60000), num=15, dtype=int).tolist()
    n = len(data)
    
    # x_lims = [-0.5, 1.5]

    sel_ids = [0, 2*n//3, n-1] 
    plt.figure(figsize=(10, 6))
    for i in sel_ids:
        label = r"$\hat F$ for $n = $" + str(n_list[i])
        plt.plot(data[i][0, :], data[i][1, :], '-o', label=label, markersize=2, linewidth=1)
    plt.plot(data[-1][0, :], F_0(data[-1][0, :]), '--', label='F_0', color='black')
    plt.xlabel('x', fontsize=fontsize)
    plt.ylabel(r"$\hat F_n$", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.xlim(*x_lims)
    if save_plots:
        plt.savefig(os.path.join(save_dir, save_file), bbox_inches='tight')
        print(f"Saved figure to {os.path.join(save_dir, save_file)}")
    plt.show()

