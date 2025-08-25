set.seed(42)



# setup 
d  <-  3
num_timesteps  <-  100
tau_1  <-  1
# support = (-1/2, 1/2)
supp <- c(-1/2, 1/2)
alpha_0  <-  3
beta_0  <-  c(2/3, 2/3, 2/3)
theta_0  <- c(alpha_0, beta_0)
alphas_hoelder  <-  c(1/3, 1/2, 3/4)

# covariates
bound = sqrt(2 / 3)
X <- matrix(runif(d * num_timesteps, min = -bound, max = bound), nrow = d, ncol = num_timesteps)
X <- rbind(X, rep(1, ncol(X))) # add intercept

# number of epochs
num_epochs  <-  ceiling(log(num_timesteps/tau_1) + 1)

# valuation function
F_inv = function(x, alpha) {
    z <- sign(x - 0.5) * (2^(1 - alpha) * abs(x - 0.5))^(1 / alpha)
}
z_samples <- F_inv(runif(num_timesteps), alphas_hoelder[1]) # sample from the distribution
v_samples <- t(X) %*% theta_0 + z_samples


# algorithm of Bracale et al. (2025)
algo  <- function(X, d, v_samples,
                  tau_1, num_epochs, alpha, p_min, p_max, supp){

    H  <- p_max - p_min
    # nu(alpha) for alpha in (0, 1)
    nu_alpha  <- function(alpha){
        if (alpha < 0.5){
            return(1/(2 + alpha))
        } else {
            return((2 * alpha + 1)/(3 * alpha + 1))
        }
    }

    for(i in 2:num_epochs){
        tau_k <- tau_1 * 2^(i-1) # lenght of episode
        a_k <- ceiling(d^(alpha/(2 + alpha)) * tau_k^(nu_alpha(alpha))/2) # length of exploration phase
        I_k  <- (tau_k - tau_1):(tau_k - tau_1 + a_k - 1) # exporation phase theta
        II_k <- (tau_k - tau_1 + a_k):(tau_k - tau_1 + 2 * a_k - 1) # exploitation phase S_0

        # estimating theta
        x_k  <- X[, I_k] # covariates in the exploration phase for theta
        p_samples <- runif(length(I_k), min = p_min, max = p_max)
        y_t <- as.numeric(v_samples[I_k] >= p_samples)
        ols_data  <- data.frame(x_k, y_t)
        ols  <- lm(y_t ~ ., data = ols_data)
        theta_hat <- coef(ols)

        # estimating S_0
        x_k  <- X[, II_k] # covariates in the exploitation phase for S_0
        w_samples  <- runif(length(II_k), min = supp[1], max = supp[2])
        p_t  <-  theta_hat %*% t(x_k)
        
    }
}

# run the algorithm
algo(X, d, v_samples,
        tau_1, num_epochs, alphas_hoelder[1], 0 , 4, supp)
