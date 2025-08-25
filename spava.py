import numpy as np
import cvxpy as cp
import numpy.typing as npt
import torch
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve


def spav(y: npt.NDArray[np.float64], lam: float = 3.0) -> npt.NDArray[np.float64]:
    """
    Global smoothed isotonic regression using QP:
    Solves: min_x sum (y - x)^2 + lam * sum (x_{i+1} - x_i)^2
    subject to: x_1 <= x_2 <= ... <= x_n

    Parameters:
        y   : array-like, observed data
        lam : float, regularization parameter (lambda)

    Returns:
        x : array, globally smoothed and monotonic fit
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    x = cp.Variable(n)

    objective = cp.sum_squares(x - y) + lam * cp.sum_squares(x[1:] - x[:-1])
    constraints = [x[i] <= x[i+1] for i in range(n - 1)]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    return x.value




def pav(y):
    """Pool Adjacent Violators algorithm: isotonic regression."""
    y = y.clone()
    n = len(y)
    while True:
        diffs = y[1:] < y[:-1]
        if not diffs.any():
            break
        i = torch.nonzero(diffs)[0].item()
        j = i + 1
        while j < n and y[j] < y[j - 1]:
            j += 1
        avg = y[i:j].mean()
        y[i:j] = avg
    return y

def conjugate_gradient(A, b, x0=None, tol=1e-5, max_iter=1000):
    """
    Differentiable conjugate gradient solver for Ax = b.
    A: function or matrix (can be sparse), shape (n, n)
    b: tensor, shape (n,)
    x0: initial guess
    Returns: x
    """
    n = b.shape[0]
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0
    r = b - A @ x
    p = r.clone()
    rs_old = torch.dot(r, r)
    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new
    return x


def spav_pytorch_qp(y, lam=1.0, mu=1000.0, iterations=50):
    """
    Differentiable isotonic regression via quadratic programming.
    Solves: min_x sum (y - x)^2 + lam * sum (x_{i+1} - x_i)^2
    subject to: x_1 <= x_2 <= ... <= x_n
    """
    n = y.shape[0]
    device = y.device
    dtype = y.dtype

    # Construct difference matrix D for smoothness penalty
    # D = torch.eye(n - 1, n, device=device, dtype=dtype) - torch.eye(n - 1, n, device=device, dtype=dtype, requires_grad=True).roll(-1, dims=1)
    # QP: min_x 0.5 x^T Q x - b^T x, subject to Ax <= b
    # Q = torch.eye(n, device=device, dtype=dtype, requires_grad=True) + lam * D.T @ D
    # b = y

# Construct tridiagonal matrix S = D^T D as sparse
    main_diag = torch.full((n,), 2.0, device=device, dtype=dtype)
    main_diag[0] = main_diag[-1] = 1.0
    off_diag = torch.full((n - 1,), -1.0, device=device, dtype=dtype)

    indices = torch.stack([
        torch.cat([torch.arange(n), torch.arange(n - 1), torch.arange(1, n)]),
        torch.cat([torch.arange(n), torch.arange(1, n), torch.arange(n - 1)])
    ])
    values = torch.cat([main_diag, off_diag, off_diag])

    S = torch.sparse_coo_tensor(indices, values, (n, n), device=device, dtype=dtype)
    Q = torch.eye(n, device=device, dtype=dtype).to_sparse_coo() + lam * S
    Q = Q.coalesce()
    Q.requires_grad_(True)
    b = y


    # Q: torch.sparse_csr_tensor, b: torch.Tensor
    x = conjugate_gradient(Q, b)
    # x = torch.linalg.solve(Q, b)

    # For strict monotonicity, project onto monotonic cone using soft penalties
    # Add a soft penalty for violations: sum(ReLU(x[:-1] - x[1:]))^2
    mono_penalty = torch.sum(torch.relu(x[:-1] - x[1:]) ** 2)
    loss = torch.sum((x - y) ** 2) + lam * torch.sum((x[1:] - x[:-1]) ** 2) + mu * mono_penalty

    # Backpropagate through the solution
    for _ in range(iterations):
        mono_penalty = torch.sum(torch.relu(x[:-1] - x[1:]) ** 2)
        loss = torch.sum((x - y) ** 2) + lam * torch.sum((x[1:] - x[:-1]) ** 2) + mu * mono_penalty
        grad, = torch.autograd.grad(loss, x, create_graph=True)
        x = x - 1e-2 * grad

    return x



def spav_pytorch(x, y, mu=10.0, lam=10.0, lr=0.1, steps=500):
    """
    Differentiable smoothed isotonic regression in PyTorch
    """
    # if not torch.is_tensor(y):
    #     y = torch.tensor(y, dtype=torch.float32)
    # if not torch.is_tensor(x):
    #     x = torch.tensor(x, dtype=torch.float32)
    # y = y.detach() if not y.requires_grad else y
    # n = y.shape[0]
    z = y.clone().requires_grad_(True)

    log_w = torch.zeros_like(y, requires_grad=True)
    w = torch.ones_like(y)
    epsilon = 1e-8

    # opt = torch.optim.AdamW([z, log_w], lr=lr, weight_decay=1e-2)
    # opt = torch.optim.AdamW([z], lr=lr, weight_decay=1e-2)
    losses = []
    for i in range(steps):
        # Smoothness penalty: first differences
        diff_z = torch.diff(z)
        diff_x = torch.diff(x)
        if torch.isnan(z).any() or torch.isnan(diff_z).any():
            print(f"NaN detected at step {i}")
            print(f"z: {z}")
            print(f"diff_z: {diff_z}")
            break

        mu_i = torch.ones_like(diff_z)/(diff_x + epsilon)
        # mu_i = torch.ones_like(diff_z)
        # w = torch.exp(log_w)
        smooth_penalty = mu * torch.sum(mu_i * diff_z ** 2)
        data_fit = torch.sum(w * (z - y) ** 2)
        mono_penalty = torch.sum(torch.relu(-diff_z)**2)
        loss = data_fit + mu * smooth_penalty + lam * mono_penalty


        # Compute gradients
        grad, = torch.autograd.grad(loss, z, create_graph=True)
        grad = grad.clamp(-1.0, 1.0)
        z = z - lr * grad

    return z





def spav_jax(y, lam=10.0):
    """
    Smoothed isotonic regression in JAX using PGD and autodiff.
    """
    y = jnp.asarray(y)
    n = y.shape[0]

    # Construct (I + lambda * D.T @ D)
    I = jnp.eye(n)
    D = jnp.eye(n - 1, n) - jnp.eye(n - 1, n, k=1)
    Q = I + lam * (D.T @ D)

    # Solve unconstrained: Q x = y
    # x_unconstrained = solve(Q, y)
    x_unconstrained = solve(Q, y, assume_a='pos', check_finite=False)

    # Now project onto the monotonic cone via PAV
    def pav(y):
        y = y.copy()
        n = y.shape[0]
        # i = 0
        # while i < n - 1:
        #     if y[i] > y[i + 1]:
        #         j = i
        #         while j >= 0 and y[j] > y[i + 1]:
        #             j -= 1
        #         avg = jnp.mean(y[j + 1:i + 2])
        #         y = y.at[j + 1:i + 2].set(avg)
        #         i = j if j >= 0 else 0
        #     else:
        #         i += 1
        def cond_fun(state):
            i, y = state
            return i < n - 1

        def body_fun(state):
            i, y = state
            
            def inner_cond(inner_state):
                j, y = inner_state
                return (j >= 0) & (y[j] > y[i + 1])

            def inner_body(inner_state):
                j, y = inner_state
                return (j - 1, y)

            j = 1
            j, _ = jax.lax.while_loop(inner_cond, inner_body, (i, y))

            max_len = y.shape[0]
            slice_start = j + 1
            slice_size = max_len - slice_start  # static upper bound
            slice_ = jax.lax.dynamic_slice(y, (slice_start,), (slice_size,))
            valid_len = i + 2 - (j + 1)
            mask = jnp.arange(slice_size) < valid_len
            masked_slice = jnp.where(mask, slice_, 0.0)
            avg = jnp.sum(masked_slice) / valid_len
            y = jax.lax.dynamic_update_slice(y, jnp.full((valid_len,), avg), (j + 1,))
            i_new = j if j >= 0 else 0
            return (i_new, y)

        i0 = 0
        state = (i0, y)
        i_fin, y_fin = jax.lax.while_loop(cond_fun, body_fun, state)

        return y_fin

    return pav(x_unconstrained)









if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n = 50
    x_vals = np.linspace(0, 1, n)
    y_true = np.sort(np.sin(2 * np.pi * x_vals))
    y_obs = y_true + 0.15 * np.random.randn(n)

    y_fit_pav = spav(y_obs, lam=3.0)
    mu = 0.2
    y_fit = spav_pytorch_qp(torch.tensor(y_obs, dtype=torch.float32), lam=3.0)
    y_fit = y_fit.detach().numpy() if isinstance(y_fit, torch.Tensor) else y_fit

    plt.plot(x_vals, y_obs, 'o', label='Noisy Data')
    plt.plot(x_vals, y_fit, '-', label='SPAV Torch Fit')
    plt.plot(x_vals, y_fit_pav, '-', label='SPAV Fit')
    plt.plot(x_vals, y_true, '--', label='True Signal')
    plt.legend()
    plt.title("Smoothed Isotonic Regression (Vectorized SPAV)")
    plt.show()
