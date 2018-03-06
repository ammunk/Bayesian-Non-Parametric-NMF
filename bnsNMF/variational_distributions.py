import numpy as np
from scipy.stats import truncnorm, gamma, expon
from bnsNMF.datatools import trunc_moments, means_factor_prod,\
                             gamma_moments, gamma_prior_elbo, mean_sq_and_sum


class FactorTruncNormVB:

    def __init__(self, lower_bound, upper_bound, D):
        self.dist = truncnorm(lower_bound, upper_bound)
        self.D = D

    def sampling(self, n_samples, *args, **kwargs):
        self.dist.rvs(*args, size=n_samples, **kwargs)

    def moments(self):
        self.mean, self.var, self.etrpy = trunc_moments(
                                                         self.dist, 
                                                         self.sigma_sq, 
                                                         self.mu
                                                         )
        self.mean_sq, self.mean_sq_sum = mean_sq_and_sum(self.mean, self.var)

    def elbo_part(self, mean_inv_lambda, ln_mean_lambda):
        """
            Regardless whether the lambda moments are vectors (ARD)
            or matrices (psNMF), as we rely on broadcasting, when we
            deal vectors
        """

        prior_elbo = -np.sum(self.mean*mean_inv_lambda + ln_mean_lambda)
        entropy_elbo = np.sum(self.etrpy)

        return prior_elbo + entropy

class WFactor(FactorTruncNormVB):

    def update(X, mean_lambda, mean_tau, mean_H, mean_sq_sum_H):

        sigma_sq = np.reciprocal(mean_tau*mean_sq_sum_H.transpose())
        self.sigma_sq = sigma_sq.repeat(self.D, axis=1)

        factor_mix = prod_sum_W(X, self.mean, mean_H)
        self.mu = self.sigma_sq * (mean_tau*factor_mix - mean_lambda)

        self.moments()

class HFactor(FactorTruncNormVB):

    def update(X, mean_lambda, mean_tau, mean_W, mean_sq_sum_W):

        sigma_sq = np.reciprocal(mean_tau*mean_sq_sum_W)
        self.sigma_sq = sigma_sq.repeat(self.D, axis=0)

        factor_mix = prod_sum_H(X, self.mean, mean_W)
        self.mu = self.sigma_sq * (mean_tau*factor_mix - mean_lambda)

        self.moments()

class NoiseGamma:

    def __init__(self, data_size, X):
        self.data_size
        self.XT = X.transpose()
        self.tr_XX = np.sum(np.inner(self.XT, X))

    def update(mean_W, mean_sq_sum_W, mean_H,
               mean_sq_sum_H):
        mean_W_prod = means_factor_prod(mean_sq_sum_W, mean_W)
        mean_H_prod = means_factor_prod(mean_sq_sum_H,
                                        mean_H.transpose())

        # Use np.inner for fast trace of two matrices
        trace_XWH = np.sum(np.inner(self.XT, np.dot(mean_W, mean_H)))
        trace_WWHH = np.sum(np.inner(mean_W_prod, mean_H_prod))

        self.beta = self.beta + 0.5 * (self.trace_XX + trace_WWHH) \
                    - 2 * trace_XWH
        self.alpha = self.alpha + 0.5 * self.data_size

        self.moments()

    def sampling(self, n_samples, *args, **kwargs):
        self.dist = gamma(self.alpha)

    def elbo_part(self, alpha_tau, beta_tau):
        prior_elbo =  gamma_prior_elbo(self.mean, self.ln_mean, alpha_tau,
                                       beta_tau)
        entropy_elbo = self.etrpy

        return prior_elbo + entropy_elbo

    def moments(self):
        moments = gamma_moments(self.alpha, self.beta)
        self.mean, self.var, selv.inv_mean, self.ln_mean, self.etrpy = moments


class FactorPriorGamma:

    def __init__(self):

    def update(mean_sq):
        self.alpha = self.alpha + 0.5
        self.beta = self.beta + 0.5 * mean_sq

        self.moments()

    def sampling(self, n_samples, *args, **kwargs):
        self.dist = gamma(self.alpha)
        self.dist.rvs(*args, size=n_samples, **kwargs)

    def elbo_part(self, alpha_tau, beta_tau):
        prior_elbo =  np.sum(gamma_prior_elbo(self.mean, self.ln_mean, 
                             alpha_tau, beta_tau))

        entropy_elbo = np.sum(self.etrpy)

        return prior_elbo + entropy_elbo

    def moments(self):
        moments = gamma_moments(self.alpha, self.beta)
        self.mean, self.var, selv.inv_mean, self.ln_mean, self.etrpy = moments


class FactorPriorGammaARD:

    def __init__(self, data_circum):
        self.data_circum = data_circum

    def update(mean_sq_W, mean_sq_H):
        self.alpha = self.alpha + self.data_circum
        self.beta = self.beta + (np.sum(mean_sq_W, axis=1) 
                                       + np.sum(mean_sq_H, axis=0))

        self.moments()

    def sampling(self, n_samples, *args, **kwargs):
        self.dist = gamma(self.alpha)
        self.dist.rvs(*args, size=n_samples, **kwargs)

    def elbo_part(self, alpha_tau, beta_tau):

        prior_elbo =  np.sum(gamma_prior_elbo(self.mean, self.ln_mean, 
                             alpha_tau, beta_tau))

        entropy_elbo = np.sum(self.etrpy)

        return prior_elbo + entropy_elbo

    def moments(self):
        moments = gamma_moments(self.alpha, self.beta)
        self.mean, self.var, selv.inv_mean, self.ln_mean, self.etrpy = moments
