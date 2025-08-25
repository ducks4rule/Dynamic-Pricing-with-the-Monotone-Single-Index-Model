import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cma

def pava_jax(y: jnp.ndarray, decreasing: bool = False) -> jnp.ndarray:
    if decreasing:
        y = -y

    n = y.shape[0]
    y = y.astype(jnp.float32)
    weights = jnp.ones(n, dtype=jnp.float32)
    out = y.copy()
    wts = weights.copy()

    def merge_left(carry):
        out, wts, i = carry

        def cond_fun(val):
            out, wts, j = val
            return (j > 0) & (out[j-1] > out[j])

        def body_fun(val):
            out, wts, j = val
            total_wt = wts[j-1] + wts[j]
            merged_val = (wts[j-1]*out[j-1] + wts[j]*out[j]) / total_wt
            out = out.at[j-1].set(merged_val)
            out = out.at[j].set(merged_val)
            wts = wts.at[j-1].set(total_wt)
            wts = wts.at[j].set(total_wt)
            return (out, wts, j-1)

        out, wts, i = jax.lax.while_loop(cond_fun, body_fun, (out, wts, i))
        return out, wts

    def body_fun(i, state):
        out, wts = state
        out, wts = merge_left((out, wts, i))
        return (out, wts)

    out, wts = jax.lax.fori_loop(0, n, body_fun, (out, wts))

    if decreasing:
        out = -out
    return out

def pava_jax_2(y: jnp.ndarray, decreasing: bool = False) -> jnp.ndarray:
    y = y.astype(jnp.float32)
    if decreasing:
        y = -y

    n = y.shape[0]
    weights = jnp.ones(n, dtype=y.dtype)

    def body_fun(carry, i):
        y, w = carry
        cond = y[i-1] > y[i]
        avg = (w[i-1] * y[i-1] + w[i] * y[i]) / (w[i-1] + w[i])
        y = y.at[i-1].set(jnp.where(cond, avg, y[i-1]))
        y = y.at[i].set(jnp.where(cond, avg, y[i]))
        w = w.at[i-1].set(jnp.where(cond, w[i-1] + w[i], w[i-1]))
        w = w.at[i].set(jnp.where(cond, w[i-1] + w[i], w[i]))
        return (y, w), None

    def outer_cond(state):
        y, w, count = state
        return jnp.logical_and(jnp.any(y[:-1] > y[1:]), count < 1000)

    def outer_body(state):
        y, w, count = state
        # Sweep left-to-right
        (y, w), _ = jax.lax.scan(body_fun, (y, w), jnp.arange(1, n))
        # Sweep right-to-left
        (y, w), _ = jax.lax.scan(body_fun, (y, w), jnp.arange(n-1, 0, -1))
        return y, w, count + 1

    y_out, _, count = jax.lax.while_loop(outer_cond, outer_body, (y, weights, 0))

    if decreasing:
        y_out = -y_out

    return y_out


def pava_jax_stack(y: jnp.ndarray, decreasing: bool = False) -> jnp.ndarray:
    y = y.astype(jnp.float32)
    if decreasing:
        y = -y

    n = y.shape[0]
    values = y
    weights = jnp.ones(n, dtype=y.dtype)

    def scan_fn(carry, x):
        stack_vals, stack_wts, stack_len = carry
        val, wt = x
        stack_vals = stack_vals.at[stack_len].set(val)
        stack_wts = stack_wts.at[stack_len].set(wt)
        stack_len += 1

        def merge_cond(carry):
            stack_vals, stack_wts, stack_len = carry
            return jnp.logical_and(stack_len > 1, stack_vals[stack_len - 2] > stack_vals[stack_len - 1])

        def merge_body(carry):
            stack_vals, stack_wts, stack_len = carry
            w1 = stack_wts[stack_len - 2]
            w2 = stack_wts[stack_len - 1]
            v1 = stack_vals[stack_len - 2]
            v2 = stack_vals[stack_len - 1]
            new_w = w1 + w2
            new_v = (w1 * v1 + w2 * v2) / new_w
            stack_vals = stack_vals.at[stack_len - 2].set(new_v)
            stack_wts = stack_wts.at[stack_len - 2].set(new_w)
            stack_len -= 1
            return stack_vals, stack_wts, stack_len

        stack_vals, stack_wts, stack_len = jax.lax.while_loop(
            merge_cond, merge_body, (stack_vals, stack_wts, stack_len)
        )
        return (stack_vals, stack_wts, stack_len), 0

    # Preallocate stack arrays
    stack_vals = jnp.zeros(n, dtype=y.dtype)
    stack_wts = jnp.zeros(n, dtype=y.dtype)
    stack_len = jnp.array(0, dtype=jnp.int32)

    (final_vals, final_wts, final_len), _ = jax.lax.scan(
        scan_fn, (stack_vals, stack_wts, stack_len), (values, weights)
    )

    # Mask for valid blocks
    valid_mask = jnp.arange(n) < final_len
    block_sizes = jnp.where(valid_mask, final_wts, 0).astype(jnp.int32)
    block_vals = jnp.where(valid_mask, final_vals, 0.0)

    # Compute block start indices
    block_starts = jnp.cumsum(jnp.concatenate([jnp.array([0]), block_sizes[:-1]]))

    # For each position, find which block it belongs to
    idxs = jnp.arange(n)
    def find_block(i):
        # Find the last block_start <= i
        return jnp.sum(i >= block_starts) - 1
    block_idx = jax.vmap(find_block)(idxs)
    out = block_vals[block_idx]


    if decreasing:
        out = -out

    return out


def profile_loss(theta, X, y, p):
    w_t = p - jnp.dot(X, theta)
    order = jnp.argsort(w_t)
    y_sorted = y[order]
    # F_hat = pava_jax(1 - y_sorted, decreasing=False)
    F_hat = pava_jax_stack(1 - y_sorted, decreasing=False)
    loss = jnp.mean((F_hat - (1 - y_sorted))**2)
    return loss

def make_numpy_loss(jax_loss_fn, X, y, p):
    def wrapped(theta_np):
        theta = jnp.array(theta_np)
        return float(jax_loss_fn(theta, X, y, p))  # Ensure float output
    return wrapped

def optimization_profile_loss(X, y, p, lr=1e-2, steps=100):
    dim = X.shape[1]
    theta = jnp.ones(dim) / jnp.sqrt(dim)
    key = jax.random.PRNGKey(42)
    theta = jax.random.normal(key, (dim,))
    # opt = optax.adam(lr)
    opt = optax.sgd(lr, momentum=0.9)
    opt_state = opt.init(theta)

    @jax.jit
    def step_grad(theta, opt_state, step):
        loss, grads = jax.value_and_grad(profile_loss)(theta, X, y, p)
        # loss, grads = jax.grad(profile_loss, allow_int=True)(theta, X, y, p)
        # Add noise for first 10 steps
        noise_scale = jnp.where(step < 10, 0.4, 0.0)  
        grads = grads + noise_scale * jax.random.normal(jax.random.PRNGKey(step), grads.shape)
        updates, opt_state = opt.update(grads, opt_state)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss

    loss_train = jnp.zeros(steps)
    for i in range(steps):
        theta, opt_state, loss = step_grad(theta, opt_state, i)
        loss_train = loss_train.at[i].set(loss)
        if i % (steps//10) == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
            if loss_train[i] - loss_train[i-1] < 1e-6:
                print("Convergence reached.")
                break
                
    return theta, loss_train

def optimize_nelder_mead(jax_loss_fn, X, y, p):
    d = X.shape[1]
    x0 = jnp.ones(d) / jnp.sqrt(d)
    wrapped = make_numpy_loss(jax_loss_fn, X, y, p)
    res = minimize(wrapped, x0, method='Nelder-Mead')
    return jnp.array(res.x)

def optimize_cma_es(jax_loss_fn, X, y, p, sigma=0.5, steps=100):
    d = X.shape[1]
    x0 = jnp.ones(d) / jnp.sqrt(d)
    wrapped = make_numpy_loss(jax_loss_fn, X, y, p)

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma, {'maxiter': steps})
    while not es.stop():
        solutions = es.ask()
        losses = [wrapped(sol) for sol in solutions]
        es.tell(solutions, losses)
    return jnp.array(es.result.xbest)



if __name__ == "__main__":
    theta_0 = jnp.array([1, 2])
    # theta_0 = jnp.array([1, 1, 1, 3])
    # theta_0 = jnp.sqrt(2)/3 * jnp.ones(3)
    d = len(theta_0)
    n = 10000

    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (n, d))
    # X = jax.hstack((X, jnp.ones((n, 1))))  # Add intercept term

    z_samples = jax.random.uniform(key, (n,), minval=-0.5, maxval=0.5)
    # z_samples, _ = sample_bracale_fan(n)


    vt = jnp.dot(X, theta_0) + z_samples

    # pt = vt + jax.random.normal(scale=0.5, size=n)
    # pt = vt + jax.random.uniform(key, (n,), minval=-0.5, maxval=0.5)
    pt = jax.random.uniform(key, (n,), minval=0, maxval=5)
    yt = jnp.where(pt <= vt, 1, 0)
    wt = pt - jnp.dot(X, theta_0)
    order = jnp.argsort(wt)
    y_sorted = yt[order]

    # opt_theta, loss_train = optimization_profile_loss(X, yt, pt, lr=1e-4, steps=50)
    # opt_theta = optimize_nelder_mead(profile_loss, X, yt, pt)
    opt_theta = optimize_cma_es(profile_loss, X, yt, pt, sigma=0.5, steps=100)

    w_t = pt - jnp.dot(X, opt_theta)
    order_t = jnp.argsort(w_t)
    y_sorted_t = yt[order_t]
    F_hat = pava_jax_stack(1 - y_sorted_t, decreasing=False)

    # plt.plot(loss_train, label='Loss Progression')
    plt.plot(F_hat, label='F_hat', marker='o', markersize=2)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss Progression during Optimization')
    plt.legend()
    plt.show()

    print("Optimized theta:", opt_theta)


