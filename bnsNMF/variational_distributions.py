import numpy as np
from scipy.stats import truncnorm, gamma, expon
from bnsNMF.datatools import trunc_moments, means_factor_prod,\
                             gamma_moments, gamma_prior_elbo, mean_sq,\
                             mean_X_LR_error_fast


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

class LFactor(FactorTruncNormVB):

    def __init__(self, lower_bound, upper_bound, D, I):
        super().__init__(lower_bound, upper_bound, D)
        self.I = I
        self.mu = np.empty([I, D])
        self.var = self.mu.copy()
        self.etrpy = self.var.copy()

    def update(self, X, mean_lambda, mean_tau, mean_R, mean_sq_sum_R):
        self.sigma_sq = np.reciprocal(mean_tau*mean_sq_sum_R)

        RR_zero_diag = mean_R @ mean_R.transpose()
        np.fill_diagonal(RR_zero_diag, 0)
        RX = X @ mean_R.transpose() 

        for d in np.random.permutation(self.D):
            prod_mix = RX[:,d] - self.mean @ RR_zero_diag[:,d]
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

class BNPWFactor(FactorTruncNormVB):

    def __init__(self, lower_bound, upper_bound, D, I):
        super().__init__(lower_bound, upper_bound, D)
        self.I = I
        self.mu = np.empty([I, D])
        self.var = self.mu.copy()
        self.etrpy = self.var.copy()

    def update(self, X, mean_lambda, mean_tau, mean_R, mean_sq_sum_R, mean_Z,
               mean_sq_Z):
        self.sigma_sq = np.reciprocal(mean_tau*mean_sq_sum_H*mean_sq_Z)

        RR_zero_diag = mean_R @ mean_R.transpose()
        np.fill_diagonal(RR_zero_diag, 0)
        ZRX = mean_Z * (X @ mean_R.transpose())

        for d in np.random.permutation(self.D):
            prod_mix = ZRX[:,d] - mean_Z[:,d] * ((mean_Z * self.mean)
                                  @ RR_zero_diag[:,d])
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

class FactorBernoulli:

    def __init__(self, D, I):
        self.D = D
        self.pi = np.empty([I,D])
        self.var = self.pi.copy()
        self.etrpy = self.var.copy()

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

    def update(self, X, mean_pi, mean_one_ln_pi, mean_tau, mean_W, mean_sq_W,
               mean_R, mean_sq_sum_R):

        RR_zero_diag = mean_R @ mean_R.transpose()
        np.fill_diagonal(RR_zero_diag, 0)
        WRX = mean_W * (X @ mean_R.transpose()) 

        for d in np.random.permutation(self.D):
            prod_mix = WRX[:,d] - 0.5 * mean_W[:,d] * ((mean_W*self.mean) 
                                         @ RR_zero_diag[:,d])
            self.pi[:,d] =  mean_tau*(prod_mix 
                                      + 0.5 * mean_sq_W[:,d] * mean_sq_sum_R)
            self.pi[:,d] += mean_ln_pi[d] - mean_one_ln_pi[d]
            self.pi[:,d] = expit(self.pi[:,d]) # sigmoid function

            self.moments(d)

        self.cal_mean_sq_sum()

    def cal_mean_sq_sum(self):
        self.mean_sq_sum = np.sum(self.mean_sq, axis=0)

    def elbo_part(self, ln_mean_pi, ln_o_mean_pi):
        # broadcasting will be done correctly
        return np.sum(self.mean*(ln_mean_pi + ln_o_mean_pi) + ln_o_mean_pi)


    def moments(self, d):
        self.mean[:,d], self.var[:,d], self.etrpy[:,d] = bernoulli_moments(
                                                         self.pi[:,d])
        self.mean_sq[:,d] = mean_sq(self.mean[:,d], self.var[:,d])


class RFactor(FactorTruncNormVB):

    def __init__(self, lower_bound, upper_bound, D, J):
        super().__init__(lower_bound, upper_bound, D)
        self.J = J
        self.mu = np.empty([D,J])
        self.var = self.mu.copy()
        self.etrpy = self.var.copy()

    def update(self, X, mean_lambda, mean_tau, mean_L, mean_sq_sum_L):

        self.sigma_sq = np.reciprocal(mean_tau*mean_sq_sum_L)

        LL_zero_diag = mean_L.transpose() @ mean_L
        np.fill_diagonal(LL_zero_diag, 0)
        LX = mean_L.transpose() @ X

        for d in np.random.permutation(self.D):
            prod_mix =  LX[d,:] - LL_zero_diag[d,:] @ self.mean
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

    def update(self, mean_L, mean_sq_sum_L, mean_R,
               mean_sq_sum_R):

        self.sum_mean_sq_error =  mean_X_LR_error_fast(self.X, mean_L, 
                                                  mean_sq_sum_L, mean_R,
                                                  mean_sq_sum_R)
        # update beta (alpha needs only one update - see __init__())
        self.beta = self.tau_beta + 0.5 * self.sum_mean_sq_error

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

        self.alpha = lambda_alpha

    def update(self, factor_mean):
        # update beta (alpha needs only one update - see __init__())
        self.beta = self.lambda_beta + factor_mean

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

    def update(self, mean_L, mean_R):
        # update beta (alpha needs only one update - see __init__())
        self.beta = self.lambda_beta + (np.sum(mean_L, axis=0) 
                                       + np.sum(mean_R, axis=1))

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

class BetaPrior:

    def __init__(self, N, K, pi_alpha, pi_beta):
        self.N = N
        self.pi_alpha = pi_alpha
        self.pi_beta = pi_beta

    def update(self, mean_Z):
        sum_Z = np.sum(mean_Z, axis=0)
        self.alpha = self.pi_alpha + sum_Z
        self.beta = self.pi_beta + self.N - sum_Z

        self.moments()

    def elbo_part(self):
        ln_B = -(gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))
        return (self.alpha - 1)*self.ln_mean \
               + (self.beta - 1)*self.ln_o_mean_pi + ln_B

    def moments(self):
        moments = beta_moments(self.alpha, self.beta)
        self.mean, self.var, self.ln_mean, self.ln_o_mean, self.etrpy = moments
