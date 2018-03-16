import numpy as np
from scipy.special import digamma, gamma, gammaln
from scipy.stats import expon, gamma as gamma_dist
from bnsNMF.truncnorm_moments import moments

vmoments = np.vectorize(moments)
def trunc_moments(lower_bound, upper_bound, mu, sigma_sq):

    # vectorize moments(), such that moments() works elementwise when applied
    # to numpy arrays. If dimensions of the inputs does not match - numpy 
    # broadcasting is applied.
    _, _, mean, var, entropy = vmoments(lower_bound, upper_bound, mu, sigma_sq)
    return mean, var, entropy

def mean_X_WH_error(X, W_mean, W_mean_sq,
                    H_mean, H_mean_sq):

    XWH = -2*np.sum(X * (W_mean @ H_mean)) # equals -2trace(X.T @ <W> @ <H>)
    sum_var_WH= np.sum(
                        W_mean_sq @ H_mean_sq
                        - (W_mean**2 @ H_mean**2)
                        )
    return np.sum(X**2) + XWH + sum_var_WH + np.sum((W_mean @ H_mean)**2)


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

