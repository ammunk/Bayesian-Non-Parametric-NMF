import numpy as np
from scipy.special import digamma, gamma

def trunc_moments(truncnorm, sigma_sq, mu):

    mean, var = truncnorm.stats(loc=mu, scale=sigma_sq, moments='mv')
    entropy = truncnorm.entropy(loc=mu, scale=sigma_sq)

    return mean, var, entropy

def means_factor_prod(sum_sq_factor, mean_factor):
    """

    Calculates <X^TX>

    """

    mean_prod = np.dot(mean_factor.transpose(), mean_factor)
    n = mean_prod.shape()[0]
    mean_prod.flat[::n + 1] = sum_sq_factor # .flat[] allows for inplace assignment

    return mean_prod

def prod_sum_W(X, mean_W, mean_H):
    """
    Calculates W_MU = H*X^T - W_old*HH_zero_diag

    """

    HH_zero_diag = np.dot(mean_H, mean_H.transpose())
    np.fill_diagonal(HH_zero_diag, 0)

    HX = np.dot(mean_H, X.transpose())
    WHH = np.dot(mean_W, HH_zero_diag)

    return HX - WHH

def prod_sum_H(X, mean_H, mean_W):
    """
    Calculates factor_mix = W^T*X - WW_zero_diag*H_old

    """

    WW_zero_diag = np.dot(mean_W, mean_W.transpose())
    np.fill_diagonal(WW_zero_diag, 0)

    WX = np.dot(mean_W.transpose(), X)
    HWW = np.dot(WW_zero_diag, mean_H)

    return HX - WHH

def gamma_moments(alpha, beta):

    mean = alpha / beta
    var = mean / beta
    inv_mean = beta / (alpha - 1 )
    ln_mean = digamma(alpha) - np.log(beta)

    entropy = alpha - np.log(beta) + np.log(gamma(alpha)) \
              + (1 -alpha)*digamma(alpha)

    return mean, var, inv_mean, ln_mean, entropy

def gamma_factor(alpha, beta):
    return beta**alpha / gamma(alpha)

def gamma_prior_elbo(mean, ln_mean, alpha_tau, beta_tau):
    return np.log(gamma_factor(alpha_tau, beta_tau)) \
           + (alpha_tau - 1)*ln_mean \
           - beta_tau*mean

def mean_sq_and_sum(mean, var):
    """
    Calculates <XX> of stochastic variable x and sums across columns
    """
    return var + mean**2, np.inner(var + mean**2)

def nan_present(X):
    return np.isnan(np.sum(X)) # fast nan check
