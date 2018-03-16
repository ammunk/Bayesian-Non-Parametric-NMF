import numpy as np
from scipy.stats import truncnorm, gamma, expon

from bnsNMF.variational_distributions import WFactor, HFactor,\
                                             FactorPriorGamma,\
                                             FactorPriorGammaARD, NoiseGamma

from bnsNMF.loggers import Logger
from bnsNMF.datatools import sample_init_factor_element, \
                             sample_init_factor_element_ARD,\
                             mean_X_WH_error

class VBBase:

    def __init__(self, X, K, tolerance, max_iter, noise_alpha, noise_beta):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.tau_alpha = noise_alpha
        self.tau_beta = noise_beta
        self.X = X
        self.K = K
        self.I, self.J = self.X.shape
        self.data_size = self.I * self.J

        # build basic model components
        self.q_W = WFactor(0, np.infty, self.K, self.I)
        self.q_H = HFactor(0, np.infty, self.K, self.J)
        self.q_tau = NoiseGamma(self.I*self.J, self.X, 
                                self.tau_alpha, self.tau_beta)

    def train(self, run, log):

        self.run = run
        self.log = log

        ELBO_diff = np.inf
        ELBO_old = ELBO_diff
        i = 0
        self._initialize()
        while abs(ELBO_diff) > self.tolerance and i < self.max_iter:
            self._update()

            # calculate ELBO, which we seek to maximize
            # maximizing the ELBO corresponds to minimizing the 
            # KL divergence
            ELBO = self._ELBO()
            ELBO_diff = ELBO - ELBO_old
            ELBO_old = ELBO
            sum_sq_error = np.sum((self.X-self.predict())**2)
            i = i + 1
            self.run.log_scalar("training.elbo", ELBO.item(), i)
            self.run.log_scalar("training.elbo_diff", ELBO_diff.item(), i)
            self.run.log_scalar("training.sq_error", sum_sq_error.item(), i)
            msg = (f"\n\tIteration = {i}\n\tELBO = {ELBO}"
                   f"\n\tELBO diff = {ELBO_diff}\n"
                   f"\tSq Error = {sum_sq_error}\n============================"
                   "===================")
            self.log.info(msg)

    def _initialize(self):
        pass

    def _update(self):
        pass

    def predict(self):
        return self.q_W.mean @ self.q_H.mean

    def transform(self, X):
        pass

    def _ELBO(self):
        pass

    def _base_ELBO(self):
        self.tau_mean, self.ln_tau_mean = self.q_tau.mean,\
                                              self.q_tau.ln_mean

        lik_elbo = self._ELBO_likelihood()
        noise_elbo = self.q_tau.elbo_part()
        return lik_elbo + noise_elbo

    def _ELBO_likelihood(self):
        W_mean = self.q_W.mean
        W_mean_sq = self.q_W.mean_sq
        H_mean = self.q_H.mean
        H_mean_sq = self.q_H.mean_sq

        sum_mean_sq_error = mean_X_WH_error(self.X,
                                            self.q_W.mean, self.q_W.mean_sq,
                                            self.q_H.mean, self.q_H.mean_sq)
        ELBO_lik = self.data_size * 0.5 * (self.ln_tau_mean - np.log(2*np.pi)) 
        ELBO_lik = ELBO_lik - 0.5*self.tau_mean * (sum_mean_sq_error)
        return ELBO_lik

class psNMF(VBBase):

    def __init__(self, X, K, tolerance, max_iter, noise_alpha, noise_beta,
                 factor_prior_alpha, factor_prior_beta):
        super().__init__(X, K, tolerance, max_iter, noise_alpha, noise_beta)

        self.lambda_alpha = factor_prior_alpha
        self.lambda_beta = factor_prior_beta

        self.q_lambda_W = FactorPriorGamma(self.lambda_alpha, self.lambda_beta)
        self.q_lambda_H = FactorPriorGamma(self.lambda_alpha, self.lambda_beta)

    def _update(self):

        # update factors (W and H) one after the other
        self.q_W.update(self.X, self.q_lambda_W.mean, self.q_tau.mean,
                    self.q_H.mean, self.q_H.mean_sq_sum)
        self.q_H.update(self.X, self.q_lambda_H.mean, self.q_tau.mean,
                        self.q_W.mean, self.q_W.mean_sq_sum)
        # update hyperparameters (under sparse model)
        self.q_lambda_W.update(self.q_W.mean)
        self.q_lambda_H.update(self.q_H.mean)
        # noise update
        self.q_tau.update(self.q_W.mean, self.q_W.mean_sq_sum, 
                          self.q_H.mean, self.q_H.mean_sq_sum)

    def _ELBO(self):
        base_elbo = self._base_ELBO() # noise elbo is included here
        elbo_W = self.q_W.elbo_part(self.q_lambda_W.mean, 
                                    self.q_lambda_W.ln_mean)
        elbo_H = self.q_H.elbo_part(self.q_lambda_H.mean, 
                                    self.q_lambda_H.ln_mean)
        elbo_lambda_W = self.q_lambda_W.elbo_part()
        elbo_lambda_H = self.q_lambda_H.elbo_part()
        if np.any(np.isnan([base_elbo, elbo_W, elbo_H, elbo_lambda_W, elbo_lambda_H])):
            print(base_elbo, elbo_W, elbo_H, elbo_lambda_W, elbo_lambda_H)
        return base_elbo + elbo_W + elbo_H + elbo_lambda_W + elbo_lambda_H

    def _initialize(self):
        mean_W =  np.empty([self.I, self.K])
        var_W = mean_W.copy()

        mean_H = np.empty([self.K, self.J])
        var_H = mean_H.copy()


        for i in range(self.I):
            for d in range(self.K):
                estimates  = sample_init_factor_element(
                                                       self.lambda_alpha,
                                                       self.lambda_beta,
                                                       n_samples_lam=10,
                                                       n_samples_F=10)
                mean_W[i,d], var_W[i,d] = estimates

        for d in range(self.K):
            for j in range(self.J):
                estimates = sample_init_factor_element(
                                                       self.lambda_alpha,
                                                       self.lambda_beta,
                                                       n_samples_lam=10,
                                                       n_samples_F=10)
                mean_H[d,j], var_H[d,j] = estimates

        self.q_W.initialize(mean_W, var_W)
        self.q_H.initialize(mean_H, var_H)

        # update other factors once!

        # update hyperparameters (under sparse model)
        self.q_lambda_W.update(self.q_W.mean)
        self.q_lambda_H.update(self.q_H.mean)
        # noise update
        self.q_tau.update(self.q_W.mean, self.q_W.mean_sq_sum, 
                          self.q_H.mean, self.q_H.mean_sq_sum)


class pNMF(VBBase):

    def __init__(self, X, K, tolerance, max_iter, noise_alpha, noise_beta,
                 factor_prior_alpha, factor_prior_beta):
        super().__init__(X, K, tolerance, max_iter, noise_alpha, noise_beta)

        self.lambda_alpha = factor_prior_alpha
        self.lambda_beta = factor_prior_beta

        self.q_lambda = FactorPriorGammaARD(self.I + self.J, self.lambda_alpha, 
                                            self.lambda_beta)

    def _update(self):

        # update factors (W and H) one after the other
        self.q_W.update(self.X, self.q_lambda.mean.reshape(1,-1), 
                        self.q_tau.mean,self.q_H.mean, self.q_H.mean_sq_sum)
        self.q_H.update(self.X, self.q_lambda.mean.reshape(-1,1), 
                        self.q_tau.mean, self.q_W.mean, self.q_W.mean_sq_sum)
        # update hyperparameters (under ARD prior)
        self.q_lambda.update(self.q_W.mean, self.q_H.mean)
        # noise update
        self.q_tau.update(self.q_W.mean, self.q_W.mean_sq_sum, 
                          self.q_H.mean, self.q_H.mean_sq_sum)

    def _ELBO(self):
        base_elbo = self._base_ELBO() # noise elbo is included here
        elbo_W = self.q_W.elbo_part(self.q_lambda.mean.reshape(1,-1), 
                                    self.q_lambda.ln_mean.reshape(1,-1))
        elbo_H = self.q_H.elbo_part(self.q_lambda.mean.reshape(-1,1), 
                                    self.q_lambda.ln_mean.reshape(-1,1))
        elbo_lambda = self.q_lambda.elbo_part()

        return base_elbo + elbo_W + elbo_H + elbo_lambda

    def _initialize(self):
        mean_W =  np.empty([self.I, self.K])
        var_W = mean_W.copy()

        mean_H = np.empty([self.K, self.J])
        var_H = mean_H.copy()


        for d in range(self.K):
            estimates  = sample_init_factor_element_ARD(
                                                    self.lambda_alpha,
                                                    self.lambda_beta, 
                                                    self.I,
                                                    n_samples_lam=10,
                                                    n_samples_F=10
                                                    )
            mean_W[:,d], var_W[:,d] = estimates

            estimates = sample_init_factor_element_ARD(
                                                    self.lambda_alpha,
                                                    self.lambda_beta, 
                                                    self.J,
                                                    n_samples_lam=10,
                                                    n_samples_F=10
                                                    )
            mean_H[d,:], var_H[d,:] = estimates

        self.q_W.initialize(mean_W, var_W)
        self.q_H.initialize(mean_H, var_H)

        # update other factors once

        # update hyperparameters (under ARD prior)
        self.q_lambda.update(self.q_W.mean, self.q_H.mean)
        # noise update
        self.q_tau.update(self.q_W.mean, self.q_W.mean_sq_sum, 
                          self.q_H.mean, self.q_H.mean_sq_sum)
