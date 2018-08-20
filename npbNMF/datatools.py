import numpy as np
import sys
from scipy.special import digamma, gammaln, xlogy
from scipy.stats import expon, gamma as gamma_dist, \
                        beta as beta_dist, bernoulli

from npbNMF.truncated_normal_functions.truncnorm_moments import moments
from npbNMF.truncated_normal_functions.trandn import trandn

vmoments = np.vectorize(moments)
def trunc_moments(lower_bound, upper_bound, mu, sigma_sq, estimate=False):

    # vectorize moments(), such that moments() works elementwise when applied
    # to numpy arrays. If dimensions of the inputs does not match - numpy
    # broadcasting is applied.
    logZhat, Zhat, mean, var, entropy = vmoments(lower_bound, upper_bound, mu,
                                                 sigma_sq)
    if np.any(mean<0) or np.any(var<0):
        if estimate == False:
            mask = np.logical_or(mean<0, var<0)
            idx = np.where(mask)
            msg = f"\nThe expectations and variances causing issues"
            msg += "\n------------------------------------------------------\n"
            msg += ("\nThus some of TN factor expectations became negative.\n"
                      +"This is likely due to parent normal means being far"
                      +"away from the lower boundary.\n"
                      +"\t- Rerunning the program may avoid this issue OR"
                      +"\t- Allow moments to be estimated!")
            sys.exit(msg)
        else:
            n_samples = 100
            if mu.ndim == 0:
                std = np.sqrt(sigma_sq)
                samples = np.zeros((1, n_samples))
            else:
                if mu.ndim is not sigma_sq.ndim:
                    sigma_sq = np.full(mu.shape, sigma_sq)
                mask = np.logical_or(mean<0, var<0)
                idx = np.where(mask)
                std = np.sqrt(sigma_sq[idx])
                mu = mu[idx]
                samples = np.zeros((np.sum(mask), n_samples))
            for n in range(n_samples):
                samples[:,n] = truncated_sampler(mu=mu, std=std,
                                                 lower_bound=lower_bound,
                                                 upper_bound=upper_bound)
            if mu.ndim == 0:
                mean = samples.mean(axis=1)
                var = samples.var(ddof=1, axis=1)
                entropy = estimate_entropy(samples, mu, std, logZhat,
                                n_samples, axis=1)
            else:
                mean[idx] = samples.mean(axis=1)
                var[idx] = samples.var(ddof=1, axis=1)
                entropy[idx] = estimate_entropy(samples, mu, std, logZhat[idx],
                                            n_samples, axis=1)

    if np.any(mean<0) or np.any(var<0):
        sys.exit("Failure - negative means or variances")
    return mean, var, entropy

def estimate_entropy(samples, mu, std, logZhat, n_samples, axis=1):
    """ Estimate entropy

        Estimated using Markov integration:

            - entropy ~ sum(-ln(p(samples))/S

    """
    from scipy.special import logsumexp
    mu = mu.reshape(-1, 1)
    std = std.reshape(-1, 1)
    logZhat = logZhat.reshape(-1,1)
    neglogpdf = -(-0.5*np.log(np.pi*2) - np.log(std)\
             -0.5*((samples - mu)/std)**2 - logZhat)
    entropy = np.sum(neglogpdf, axis=axis)/n_samples
    return entropy

###############################################################################

""" <||X-LR||^2>


    The two functions below calculates <||X-LR||^2>

    The first is a "straight-forward calculation", while the other utilizes
    matrix matrix products and traces.

"""

def mean_X_LR_error(X, L_mean, L_mean_sq, R_mean, R_mean_sq):

    XLR = -2*np.sum(X * (L_mean @ R_mean)) # equals -2trace(X.T @ <L> @ <R>)
    sum_var_LR= np.sum(
                        L_mean_sq @ R_mean_sq
                        - (L_mean**2 @ R_mean**2)
                        )
    return np.sum(X**2) + XLR + sum_var_LR + np.sum((L_mean @ R_mean)**2)

def mean_X_LR_error_fast(X, mean_L, mean_sq_sum_L, mean_R, mean_sq_sum_R):
        mean_L_prod = means_factor_prod(mean_sq_sum_L, mean_L)
        mean_R_prod = means_factor_prod(mean_sq_sum_R, mean_R.transpose())

        trace_XTLR = X.ravel() @ (mean_L @ mean_R).ravel()
        trace_LLRR = mean_L_prod.transpose().ravel() @ mean_R_prod.ravel()

        trace_XX = X.ravel() @ X.ravel()
        return trace_XX + trace_LLRR - 2*trace_XTLR

def means_factor_prod(sum_sq_factor, mean_factor):
    """

    Calculates <X^TX>

    """

    mean_prod = mean_factor.T @ mean_factor
    n = mean_prod.shape[0]
    mean_prod.flat[::n + 1] = sum_sq_factor # .flat[] allows for inplace assignment

    return mean_prod

###############################################################################

"""
Single element factor update - for calculation validation

"""

def single_element_update_W(X, tau_mean, H_mean_sq_sum, H_mean, lambda_W,
                            W_mean, Z, k, n):
    K = H_mean.shape[1]
    sigma_sq = 1/(tau_mean * H_mean_sq_sum[k]*Z[k,n] + lambda_W[k,n])
    sum_prod = 0
    for i in range(K):
        if i != k:
            sum_prod += W_mean[i,n]*Z[i,n]*(H_mean[:,k].T @ H_mean[:,i])

    mu = sigma_sq*(tau_mean*(Z[k,n]*(H_mean[:,k].T @ X[:,n])
                   - Z[k,n]*sum_prod))
    prod_mix = (Z[k,n]*(H_mean[:,k].T @ X[:,n]) - Z[k,n]*sum_prod)
    zhx =Z[k,n]*(H_mean[:,k].T @ X[:,n])

    return mu, sigma_sq, prod_mix, zhx

def single_element_update_H(X, tau_mean, WZ_mean_sq_sum, WZ_mean, lambda_H,
                            H_mean, k, d):
    K = WZ_mean.shape[0]
    sigma_sq = 1/(tau_mean * WZ_mean_sq_sum[k])
    sum_prod = 0
    for i in range(K):
        if i != k:
            sum_prod += H_mean[d,i]*WZ_mean[k,:] @ WZ_mean[i,:].T

    mu = sigma_sq*(tau_mean*((WZ_mean[k,:] @ X[d,:].T) - sum_prod)
                             - lambda_H[d,k])
    prod_mix = (WZ_mean[k,:] @ X[d,:].T) - sum_prod

    return mu, prod_mix

###############################################################################

def gamma_moments(alpha, beta):

    mean = alpha/beta
    var = mean/beta
    ln_mean = digamma(alpha) - np.log(beta)

    entropy = alpha - np.log(beta) + gammaln(alpha) \
              + (1 - alpha)*digamma(alpha)

    return mean, var, ln_mean, entropy

def gamma_prior_elbo(mean, ln_mean, alpha, beta):
    return alpha*np.log(beta) \
           - gammaln(alpha)\
           + (alpha - 1)*ln_mean \
           - beta*mean

def bernoulli_moments(pi):
    q = 1 - pi

    return pi, pi*q, -(np.nan_to_num(xlogy(q,q)) + np.nan_to_num(xlogy(pi,pi)))

def beta_moments(alpha, beta):
    mean = alpha/(alpha + beta)
    var = alpha*beta/((alpha + beta + 1)*(alpha + beta)**2)
    ln_mean = digamma(alpha) - digamma(alpha + beta)
    ln_o_mean = digamma(beta) - digamma(alpha + beta)
    entropy = (gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))\
              - (alpha - 1)*digamma(alpha) - (beta - 1)*digamma(beta)\
              + (alpha + beta - 2)*digamma(alpha + beta)
    return mean, var, ln_mean, ln_o_mean, entropy

def expon_moments(lambdas):

    mean = np.reciprocal(lambdas)
    var = mean**2
    entropy = 1 - np.log(lambdas)
    return mean, var, entropy

def mean_sq(mean, var):
    """
    Calculates <XX> of stochastic variable X
    """
    return var + mean**2

def nan_present(X):
    return np.isnan(np.sum(X)) # fast nan check

def sample_init_factor_TN(lower_bound, upper_bound, std, n_samples=10):
    # gam = gamma_dist(a=lambda_alpha, loc=0, scale=1/lambda_beta)
    # lambdas = gam.rvs(size=n_samples)

    elements = np.zeros(n_samples)
    for i in range(n_samples):
        elements[i] = truncated_sampler(0, std, lower_bound, upper_bound)
    mean = elements.mean()
    # unbiased variance
    var = elements.var(ddof=1)

    return mean, var

def sample_init_factor_expon(expected_param, n_samples=10):
    # gam = gamma_dist(a=lambda_alpha, loc=0, scale=1/lambda_beta)
    # lambdas = gam.rvs(size=n_samples)

    elements = np.zeros(n_samples)
    for i in range(n_samples):
        elements[i] = expon.rvs(loc=0, scale=1/expected_param)
    mean = elements.mean()
    # unbiased variance
    var = elements.var(ddof=1)

    return mean, var

def sample_init_Z_bernoulli(alpha, beta, n_samples=10):

    pbeta = beta_dist(a=alpha, b=beta)
    pi = pbeta.rvs(size=n_samples)

    elements = np.zeros(n_samples)
    for i, p in enumerate(pi):
        elements[i] = bernoulli.rvs(p)
    mean = elements.mean()
    # unbiased variance
    var = elements.var(ddof=1)
    return mean, var

#############################################################################

# Functions for Gibbs sampler

#############################################################################

def Nth_harmonic(beta, N):

    beta_sum = 0
    for n in range(N):
        # because python is zero indexed n starts at 0 and ends at N-1.
        # To account for this we do not subtract by 1 in the denominator
        beta_sum += 1/(beta + n)

    beta_sum *= beta

    return beta_sum


def beta_factor_logproduct(beta, K_plus, N, m):
    logprod = 0
    for k in range(K_plus):
        logprod += gammaln(m[k]) + gammaln(N - m[k] + beta) - gammaln(N + beta)
    return logprod

def logr_z_kn(m, z_tilde, w_kn, H, x_n, noise, beta, N, mask):
    logr_p = np.log(m) - np.log(beta + N - 1 - m)
    h_k = H[:,~mask]
    H_tilde = H[:,mask]
    if len(mask) > 1:
        other_features_product = np.dot(H_tilde, z_tilde[mask]).T
    else:
        other_features_product = h_k.T*0

    logr_l = -noise*((other_features_product - x_n.T
                     + 0.5*(h_k.T*w_kn)) @ (h_k*w_kn))
    return logr_p + logr_l

def truncated_sampler(mu=np.zeros(1), std=np.ones(1), lower_bound=0,
                      upper_bound=np.infty):
    samples = trandn((lower_bound-mu)/std, (upper_bound-mu)/std)
    samples = samples*std + mu
    return samples


#############################################################################

# pdf calculations

#############################################################################

def trunc_norm_sum_log_pdf(X, precision):
    logpdf = (-0.5*np.log(np.pi*2) + np.log(precision)\
              -0.5*precision*(X**2) + np.log(2))
    return np.sum(logpdf)

def gamma_sum_log_pdf(X, a, b):
    logpdf = a*np.log(b) - gammaln(a) + (a-1)*np.log(X) - b*X
    return np.sum(logpdf)

def IBP_sum_log_pdf(Z, alpha, beta, K_plus, M, N):
    log_pdf = np.log(alpha) - gammaln(N + beta)*K_plus
    if len(M) != K_plus:
        sys.exit("M too large")
    for m in M:
        log_pdf += gammaln(m) + gammaln(N - m + beta)
    log_pdf += K_plus*(np.log(alpha) + np.log(beta))
    # efficient way of calculating histories (counting unique columns)
    # - https://stackoverflow.com/questions/27000092/count-how-many-times-each-row-is-present-in-numpy-array
    # the methods really counts unique rows, but by tranposing Z, we get unique
    # columns
    dt = np.dtype((np.void, Z.T.dtype.itemsize*Z.T.shape[1]))
    b = np.ascontiguousarray(Z.T).view(dt)
    _, count = np.unique(b, return_counts=True)
    for K_h in count:
        log_pdf += gammaln(K_h+1) # use that x! = gamma(x+1)
    return log_pdf
