import numpy as np
from scipy.special import digamma, gamma, gammaln
from scipy.stats import expon, gamma as gamma_dist, \
                        beta as beta_dist, bernoulli

from bnsNMF.truncnorm_moments import moments

vmoments = np.vectorize(moments)
def trunc_moments(lower_bound, upper_bound, mu, sigma_sq):

    # vectorize moments(), such that moments() works elementwise when applied
    # to numpy arrays. If dimensions of the inputs does not match - numpy 
    # broadcasting is applied.
    _, _, mean, var, entropy = vmoments(lower_bound, upper_bound, mu, sigma_sq)
    return mean, var, entropy

###############################################################################

""" <||X-LR||^2>


    The two functions below calculates <||X-LR||^2>

    The first is a "straight-forward calculation", while the other utilizes 
    matrix matrix products and traces.

"""



def mean_X_LR_error(X, L_mean, L_mean_sq,
                    R_mean, R_mean_sq):

    XLR = -2*np.sum(X * (L_mean @ R_mean)) # equals -2trace(X.T @ <L> @ <R>)
    sum_var_LR= np.sum(
                        L_mean_sq @ R_mean_sq
                        - (L_mean**2 @ R_mean**2)
                        )
    return np.sum(X**2) + XLR + sum_var_LR + np.sum((L_mean @ R_mean)**2)

def mean_X_LR_error_fast(X, mean_L, mean_sq_sum_L,
                    mean_R, mean_sq_sum_R):
        mean_L_prod = means_factor_prod(mean_sq_sum_L, mean_L)
        mean_R_prod = means_factor_prod(mean_sq_sum_R, mean_R.transpose())

        trace_XTLR = X.ravel() @ (mean_L @ mean_R).ravel()
        trace_LLRR = mean_L_prod.transpose().ravel() @ mean_R_prod.ravel()

        trace_XX = X.ravel() @ X.ravel()
        return trace_XX + trace_LLRR - 2 * trace_XTLR

###############################################################################

def means_factor_prod(sum_sq_factor, mean_factor):
    """

    Calculates <X^TX>

    """

    mean_prod = mean_factor.transpose() @ mean_factor
    n = mean_prod.shape[0]
    mean_prod.flat[::n + 1] = sum_sq_factor # .flat[] allows for inplace assignment

    return mean_prod

def gamma_moments(alpha, beta):

    # all alphas ought to be larger than two, otherwise inv_mean is
    # ill-defined. However, we do not need it.

    # inv_mean = beta / (alpha - 1)

    mean = alpha / beta
    var = mean / beta
    ln_mean = digamma(alpha) - np.log(beta)

    entropy = alpha - np.log(beta) + gammaln(alpha) \
              + (1 - alpha)*digamma(alpha)

    return mean, var, ln_mean, entropy

def gamma_prior_elbo(mean, ln_mean, alpha_tau, beta_tau):
    return alpha_tau * np.log(beta_tau) \
           - gammaln(alpha_tau)\
           + (alpha_tau - 1)*ln_mean \
           - beta_tau*mean

def bernoulli_moments(pi):
    q = 1 - pi
    return pi, pi * q, -q*np.log(q) - p*np.log(p)

def beta_moments(alpha, beta):
    mean = alpha/(alpha + beta)
    var = alpha*beta/((alpha + beta + 1)*(alpha + beta)**2)
    ln_mean = digamma(alpha) - digamma(alpha + beta)
    ln_o_mean = digamma(beta) - digamma(alpha + beta)
    entropy = -(gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))\
              - (alpha - 1)*digamma(alpha) - (beta - 1)*digamma(beta)\
              + (alpha + beta - 2)*digamma(alpha + beta)
    return mean, var, ln_mean, ln_o_mean, entropy

def mean_sq(mean, var):
    """
    Calculates <XX> of stochastic variable x
    """
    return var + mean**2

def nan_present(X):
    return np.isnan(np.sum(X)) # fast nan check

def sample_init_factor_element(lambda_alpha, lambda_beta, 
                               n_samples_lam=50, n_samples_F=50):
    gam = gamma_dist(a=lambda_alpha, loc=0, scale=1/lambda_beta)
    lambdas = gam.rvs(size=n_samples_lam)

    elements = []
    for lam in lambdas:
        elements = np.append(elements, 
                             expon.rvs(loc=0, scale=1/lam, size=n_samples_F))
    mean = elements.mean()
    # unbiased variance
    var = elements.var(ddof=1)

    return mean, var

def sample_init_factor_element_ARD(lambda_alpha, lambda_beta, factor_dimension,
                                   n_samples_lam=50, n_samples_F=50):

    gam = gamma_dist(a=lambda_alpha, loc=0, scale=1/lambda_beta)
    lambdas = gam.rvs(size=n_samples_lam)
    samples = n_samples_F*factor_dimension

    elements = []
    for lam in lambdas:
        elements = np.append(elements, 
                         expon.rvs(loc=0, scale=1/lam, size=samples))
    elements = elements.reshape(factor_dimension, -1)
    mean = elements.mean(axis=1)
    # unbiased variance
    var = elements.var(ddof=1, axis=1)

    return mean, var
def sample_init_Z_element(pi_alpha, pi_beta, factor_dimension,
                          n_samples_pi=50, n_samples_Z=10):

    beta = beta_dist(a = pi_alpha, b = pi_beta)
    pi = beta.rvs(size=n_samples_pi)
    samples = n_samples_Z*factor_dimension

    elements = []
    for p in pi:
        elements = np.append(elements, bernoulli.rvs(p, size=samples))
    elements = elements.reshape(factor_dimension, -1)
    mean = elements.mean(axis=1)
    # unbiased variance
    var = elements.var(ddof=1, axis=1)

    return mean, var
