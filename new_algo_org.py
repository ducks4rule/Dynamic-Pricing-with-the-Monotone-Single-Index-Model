import numpy as np
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from scipy.stats import entropy

from densities import sample_bracale_fan
from new_algo_1 import grid_search

theta_0 = np.array([1, 2])
# theta_0 = np.array([1, 1, 1, 3])
# theta_0 = np.sqrt(2)/3 * np.ones(3)

# n = 10000
n = 1000
X = np.random.normal(loc=1, scale=1, size=(n, len(theta_0)))
# X = np.random.chisquare(df=2, size=(n, len(theta_0)))
# X = np.random.uniform(low=-1, high=1, size=(n, len(theta_0)))
# X = np.hstack((X, np.ones((n, 1))))  # Add intercept term

z_samples = np.random.uniform(low=-0.5, high=0.5, size=n)
# z_samples, _ = sample_bracale_fan(n)


vt = np.dot(X, theta_0) + z_samples

pt = vt + np.random.normal(scale=0.5, size=n)

y_t = np.where(pt <= vt, 1, 0)

def fit_F_hat(p, v_t, X, y, theta):
    w = p - np.dot(X, theta)
    w_order = np.argsort(w)

    F_model = IsotonicRegression(increasing=True)
    F_hat = F_model.fit_transform(w[w_order], 1 - y[w_order])
    return F_hat, w, w_order




# Suppose we know the truth theta0
# theta_test = np.array([100, 20])
theta_test = theta_0
F_hat, w_t, order = fit_F_hat(pt, vt, X, y_t, theta_test)

plt.plot(np.sort(w_t), F_hat, 'o', markersize=1, label="F_hat")
plt.plot(np.sort(w_t), 1 - y_t[order], 'o', markersize=1, label="y_t")
plt.xlabel("w_t values")
plt.ylabel("F_hat and y_t values")
plt.legend()
plt.show()


theta_hat, F_hat_vals, error_vals = grid_search(X, pt, y_t, n_points=10, run_depth=2,
                                    # bounds=[[-10, 10], [-10, 10], [-10, 10]],
                                    bounds=[[-5, 5], [-5, 5]],
                                    verbose=True)
                                    # bounds=[[-10, 10], [-10, 10], [-10, 10], [-10, 10]],



grid, errs = error_vals
grid, errs = np.array(grid), np.array(errs)


def criterion(F, y):
    # return np.linalg.norm(F_hat - (1 - y_t)) ** 2
    return np.sum((F - (1 - y)) ** 2)/len(F)

print(criterion(F_hat, y_t[order]))
print(f"theta_hat = {theta_hat} -- theta_0 = {theta_0}")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# grid: shape (2, N)
x = grid[:, 0]
y = grid[:, 1]
z = errs

# ax.scatter(x, y, z, c=z, cmap='viridis')
ax.plot_trisurf(x, y, z, cmap='viridis', alpha=0.5)
ax.set_xlabel('Theta 1')
ax.set_ylabel('Theta 2')
ax.set_zlabel('Error')
plt.show()

if False:
    crit = []
    crit_2 = []
    theta_grid = []

    for i in range(-10, 11):
        for j in range(-10, 11):
            for k in range(-10, 11):
                for l in range(-10, 11):
                    theta_ij = np.array([i, j, k, l])
                    theta_grid.append(theta_ij)
                    F_hat_ij, w_t_ij, order_ij = fit_F_hat(pt, vt, X, y_t, theta_ij)
                    crit_ij = criterion(F_hat_ij, y_t[order_ij])
                    crit.append(crit_ij)

            # plt.plot(np.sort(w_t_ij), (F_hat_ij - (1 - y_t[order_ij]))**2, 'o', markersize=1, label=f"theta_ij = {theta_ij}, \n entro = {crit_2:.2f}, crit = {crit_ij:.2f}")
            # plt.legend()
            # plt.show()


    min_index = np.argmin(crit)
    Theta_min = theta_grid[min_index]
    print(f"Theta[{min_index}] = {Theta_min}")
    print(f"Theta_hat = {theta_hat}")

    plt.plot(range(1, len(crit) + 1), crit, '-o', markersize=2)
    plt.title("Criterion values for different theta combinations")
    plt.suptitle(f"Minimum criterion value at index {min_index} with theta = {Theta_min}", fontsize=10)
    plt.xlabel("Index of theta grid")
    plt.ylabel("Criterion value")
    plt.show()


plt.plot(F_hat_vals[0], F_hat_vals[1], 'o', markersize=1, label="F_hat from grid search")
plt.show()
