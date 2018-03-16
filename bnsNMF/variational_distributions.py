import numpy as np
from scipy.stats import truncnorm, gamma, expon
from bnsNMF.datatools import trunc_moments, means_factor_prod,\
                             gamma_moments, gamma_prior_elbo, mean_sq


class FactorTruncNormVB:

    def __init__(self, lower_bound, upper_bound, D):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.D = D

    def sampling(self, n_samples, *args, **kwargs):
        self.dist.rvs(*args, size=n_samples, **kwargs)

    def elbo_part(self, mean_lambda, ln_mean_lambda):
        """ Calculate ELBO contribution from factors

            Regardless whether the lambda moments are vectors (ARD)
            or matrices (psNMF), as we rely on broadcasting, when we
            deal vectors
        """

        prior_elbo = np.sum(ln_mean_lambda - self.mean*mean_lambda)
        entropy_elbo = np.sum(self.etrpy)

        return prior_elbo + entropy_elbo

    def initialize(self, est_mean, est_var):
        """ Initialize mean and variance

        Takes estimated mean and variance as argument and set these
        as initial values.

        These values are estimated through sampling using prior
        hyperparameters
        """
        self.mean = est_mean
        self.var = est_var
        self.mean_sq = mean_sq(self.mean, self.var)
        self.cal_mean_sq_sum()

    def cal_mean_sq_sum(self):
        pass

class WFactor(FactorTruncNormVB):

    def __init__(self, lower_bound, upper_bound, D, I):
        super().__init__(lower_bound, upper_bound, D)
        self.I = I
        self.mu = np.empty([I, D])
        self.var = self.mu.copy()
        self.etrpy = self.var.copy()

    def update(self, X, mean_lambda, mean_tau, mean_H, mean_sq_sum_H):
        self.sigma_sq = np.reciprocal(mean_tau*mean_sq_sum_H)

        HH_zero_diag = mean_H @ mean_H.transpose()
        np.fill_diagonal(HH_zero_diag, 0)
        HX = X @ mean_H.transpose() 

        for d in np.random.permutation(self.D):
            prod_mix = HX[:,d] - self.mean @ HH_zero_diag[:,d]
            self.mu[:,d] =  self.sigma_sq[d] * (mean_tau * prod_mix 
                                                  - mean_lambda[:,d])
            self.moments(d)

        self.cal_mean_sq_sum()

    def cal_mean_sq_sum(self):
        self.mean_sq_sum = np.sum(self.mean_sq, axis=0)

    def moments(self, d):
        self.mean[:,d], self.var[:,d], self.etrpy[:,d] = trunc_moments(
                                                         self.lower_bound,
                                                         self.upper_bound,
                                                         self.mu[:,d],
                                                         self.sigma_sq[d]
                                                                      )
        self.mean_sq[:,d] = mean_sq(self.mean[:,d], self.var[:,d])

class HFactor(FactorTruncNormVB):

    def __init__(self, lower_bound, upper_bound, D, J):
        super().__init__(lower_bound, upper_bound, D)
        self.J = J
        self.mu = np.empty([D,J])
        self.var = self.mu.copy()
        self.etrpy = self.var.copy()

    def update(self, X, mean_lambda, mean_tau, mean_W, mean_sq_sum_W):

        self.sigma_sq = np.reciprocal(mean_tau*mean_sq_sum_W)

        WW_zero_diag = mean_W.transpose() @ mean_W
        np.fill_diagonal(WW_zero_diag, 0)
        WX = mean_W.transpose() @ X

        for d in np.random.permutation(self.D):
            prod_mix =  WX[d,:] - WW_zero_diag[d,:] @ self.mean
            self.mu[d,:] =  self.sigma_sq[d] * (mean_tau * prod_mix 
                                                  - mean_lambda[d,:])
            self.moments(d)

        self.cal_mean_sq_sum()

    def cal_mean_sq_sum(self):
        self.mean_sq_sum = np.sum(self.mean_sq, axis=1)

    def moments(self,d):
        self.mean[d,:], self.var[d,:], self.etrpy[d,:] = trunc_moments(
                                                         self.lower_bound,
                                                         self.upper_bound,
                                                         self.mu[d,:],
                                                         self.sigma_sq[d]
                                                        )
        self.mean_sq[d,:] = mean_sq(self.mean[d,:], self.var[d,:])

class NoiseGamma:

    def __init__(self, data_size, X, tau_alpha, tau_beta):
        self.tau_alpha = tau_alpha
        self.tau_beta = tau_beta
        self.X = X
        self.trace_XX = X.ravel() @ X.ravel()

        self.alpha = tau_alpha + 0.5 * data_size

    def update(self, mean_W, mean_sq_sum_W, mean_H,
               mean_sq_sum_H):
        mean_W_prod = means_factor_prod(mean_sq_sum_W, mean_W)
        mean_H_prod = means_factor_prod(mean_sq_sum_H,
                                        mean_H.transpose())

        trace_XTWH = self.X.ravel() @ (mean_W @ mean_H).ravel()
        trace_WWHH = mean_W_prod.transpose().ravel() @ mean_H_prod.ravel()

        # update beta (alpha needs only one update - see __init__())
        self.beta = self.tau_beta + 0.5 * (self.trace_XX + trace_WWHH \
                    - 2 * trace_XTWH)

        self.moments()

    def sampling(self, n_samples, *args, **kwargs):
        self.dist = gamma(self.alpha)

    def elbo_part(self):
        prior_elbo =  gamma_prior_elbo(self.mean, self.ln_mean, self.tau_alpha,
                                       self.tau_beta)
        entropy_elbo = self.etrpy

        return prior_elbo + entropy_elbo

    def moments(self):
        moments = gamma_moments(self.alpha, self.beta)
        self.mean, self.var, self.ln_mean, self.etrpy = moments

class FactorPriorGamma:

    def __init__(self, lambda_alpha, lambda_beta):

        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta

        self.alpha = lambda_alpha + 0.5

    def update(self, mean):
        # update beta (alpha needs only one update - see __init__())
        self.beta = self.lambda_beta + mean

        self.moments()

    def sampling(self, n_samples, *args, **kwargs):
        self.dist = gamma(self.alpha)
        self.dist.rvs(*args, size=n_samples, **kwargs)

    def elbo_part(self):
        prior_elbo =  np.sum(gamma_prior_elbo(self.mean, self.ln_mean, 
                             self.lambda_alpha, self.lambda_beta))

        entropy_elbo = np.sum(self.etrpy)

        return prior_elbo + entropy_elbo

    def moments(self):
        moments = gamma_moments(self.alpha, self.beta)
        self.mean, self.var, self.ln_mean, self.etrpy = moments


class FactorPriorGammaARD:

    def __init__(self, data_circum, lambda_alpha, lambda_beta):
        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta

        self.alpha = lambda_alpha + data_circum

    def update(self, mean_W, mean_H):
        # update beta (alpha needs only one update - see __init__())
        self.beta = self.lambda_beta + (np.sum(mean_W, axis=0) 
                                       + np.sum(mean_H, axis=1))

        self.moments()

    def sampling(self, n_samples, *args, **kwargs):
        self.dist = gamma(self.alpha)
        self.dist.rvs(*args, size=n_samples, **kwargs)

    def elbo_part(self):

        prior_elbo =  np.sum(gamma_prior_elbo(self.mean, self.ln_mean, 
                             self.lambda_alpha, self.lambda_beta))

        entropy_elbo = np.sum(self.etrpy)

        return prior_elbo + entropy_elbo

    def moments(self):
        moments = gamma_moments(self.alpha, self.beta)
        self.mean, self.var, self.ln_mean, self.etrpy = moments
