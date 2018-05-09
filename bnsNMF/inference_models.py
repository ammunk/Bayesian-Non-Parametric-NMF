import numpy as np
from scipy.stats import truncnorm, gamma, expon

from bnsNMF.variational_distributions import LFactor, RFactor,\
                                             FactorPriorGamma,\
                                             FactorPriorGammaARD, NoiseGamma,\
                                             BNPWFactor, BetaPrior,\
                                             FactorBernoulli,\
                                             FactorPriorBNP

from bnsNMF.loggers import Logger
from bnsNMF.datatools import sample_init_factor_element, \
                             sample_init_factor_element_ARD, \
                             sample_init_Z_element, \
                             mean_sq, mean_X_LR_error_fast


class FactorStats:

    def __init__(self, Factor, factor_type):
        self.factor = Factor # pass by reference (when Factor changes
                             # in main program, it does here too)
        self.add_factor = False
        if factor_type == "L":
            self.sum_axis = 0
        else:
            self.sum_axis = 1

    def update(self):

        if not self.add_factor:
            tmp = mean_sq(self.factor.mean, self.factor.var)
            self.mean = self.factor.mean
            self.mean_sq_sum = np.sum(tmp, axis=self.sum_axis)
        else:
            mean, added_mean = self.factor.mean, self.added_factor.mean
            var, added_var = self.factor.var, self.added_factor.var
            tmp, added_tmp = mean_sq(mean, var), mean_sq(added_mean, added_var)
            self.mean = mean*added_mean
            self.mean_sq_sum = np.sum(tmp*added_tmp, axis=self.sum_axis)

    def new_factor(self, Add_Factor):

        self.add_factor = True
        self.added_factor = Add_Factor


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
        self.q_W = LFactor(0, np.infty, self.K, self.I)
        self.q_H = RFactor(0, np.infty, self.K, self.J)
        self.stats_L = FactorStats(self.q_W, 'L')
        self.stats_R = FactorStats(self.q_H, 'R')
        self.q_tau = NoiseGamma(self.I*self.J, self.X,
                                self.tau_alpha, self.tau_beta)

    def train(self, run, log):

        self.run = run
        self.log = log

        rel_ELBO = np.inf
        ELBO_old = -1e21
        i = 0
        self.log.info("Initializing")
        self._initialize()
        while rel_ELBO > self.tolerance and i < self.max_iter:
            self._update()

            # calculate ELBO, which we seek to maximize
            # maximizing the ELBO corresponds to minimizing the
            # KL divergence
            ELBO = self._ELBO()
            ELBO_diff = ELBO - ELBO_old
            rel_ELBO = ELBO_diff / abs(ELBO_old)
            ELBO_old = ELBO
            sum_sq_error = np.sum((self.X-self.predict())**2)
            i = i + 1
            self.run.log_scalar("training.elbo", ELBO.item(), i)
            self.run.log_scalar("training.elbo_diff", rel_ELBO.item(), i)
            self.run.log_scalar("training.sq_error", sum_sq_error.item(), i)
            msg = (f"\n\tIteration = {i}\n\tELBO = {ELBO}"
                   f"\n\tRelative ELBO = {rel_ELBO}\n"
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

    def extract_features(self):
        return self.q_H.mean

    def _base_ELBO(self):
        self.tau_mean, self.ln_tau_mean = self.q_tau.mean,\
                                              self.q_tau.ln_mean

        lik_elbo = self._ELBO_likelihood()
        noise_elbo = self.q_tau.elbo_part()
        return lik_elbo + noise_elbo

    def _ELBO_likelihood(self):

        sum_mean_sq_error = self.q_tau.sum_mean_sq_error
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
                    self.q_H.mean, self.stats_R.mean_sq_sum)
        self.stats_L.update()
        self.q_H.update(self.X, self.q_lambda_H.mean, self.q_tau.mean,
                        self.q_W.mean, self.stats_L.mean_sq_sum)
        self.stats_R.update()
        # update hyperparameters (under sparse model)
        self.q_lambda_W.update(self.q_W.mean)
        self.q_lambda_H.update(self.q_H.mean)
        # noise update
        self.q_tau.update(self.q_W.mean, self.stats_L.mean_sq_sum,
                          self.q_H.mean, self.stats_R.mean_sq_sum)

    def _ELBO(self):
        base_elbo = self._base_ELBO() # noise elbo is included here
        elbo_W = self.q_W.elbo_part(self.q_lambda_W.mean,
                                    self.q_lambda_W.ln_mean)
        elbo_H = self.q_H.elbo_part(self.q_lambda_H.mean,
                                    self.q_lambda_H.ln_mean)
        elbo_lambda_W = self.q_lambda_W.elbo_part()
        elbo_lambda_H = self.q_lambda_H.elbo_part()
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
        self.stats_L.update()
        self.stats_R.update()
        # noise update
        self.q_tau.update(self.q_W.mean, self.stats_L.mean_sq_sum,
                          self.q_H.mean, self.stats_R.mean_sq_sum)


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
                        self.q_tau.mean,self.q_H.mean,
                        self.stats_R.mean_sq_sum)
        self.stats_L.update()
        self.q_H.update(self.X, self.q_lambda.mean.reshape(-1,1),
                        self.q_tau.mean, self.q_W.mean,
                        self.stats_L.mean_sq_sum)
        self.stats_R.update()
        # update hyperparameters (under ARD prior)
        self.q_lambda.update(self.q_W.mean, self.q_H.mean)
        # noise update
        self.q_tau.update(self.q_W.mean, self.stats_L.mean_sq_sum,
                          self.q_H.mean, self.stats_R.mean_sq_sum)

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
        self.stats_L.update()
        self.stats_R.update()

        # update other factors once

        # update hyperparameters (under ARD prior)
        self.q_lambda.update(self.q_W.mean, self.q_H.mean)
        # noise update
        self.q_tau.update(self.q_W.mean, self.stats_L.mean_sq_sum,
                          self.q_H.mean, self.stats_R.mean_sq_sum)

class bnsNMF(VBBase):

    def __init__(self, X, K, tolerance, max_iter, noise_alpha, noise_beta,
                 factor_prior_alpha, factor_prior_beta, pi_a, pi_b):
        super().__init__(X, K, tolerance, max_iter, noise_alpha, noise_beta)

        # add Z factor VB and overwrite Base Distributions
        self.q_W = BNPWFactor(0, np.inf, K, self.I)
        self.q_Z = FactorBernoulli(K, self.I)
        self.stats_L = FactorStats(self.q_W, 'L')
        self.stats_L.new_factor(self.q_Z)

        self.lambda_alpha = factor_prior_alpha
        self.lambda_beta = factor_prior_beta
        self.pi_alpha = pi_a/K
        self.pi_beta = pi_b*(1 - 1/K)

        self.q_lambda_W = FactorPriorBNP(self.lambda_alpha, self.lambda_beta)
        self.q_lambda_H = FactorPriorGamma(self.lambda_alpha, self.lambda_beta)
        self.q_pi = BetaPrior(self.I, self.K, self.pi_alpha, self.pi_beta)

    def _update(self):

        # update factors (W and H) one after the other
        self.q_W.update(self.X, self.q_lambda_W.mean, self.q_tau.mean,
                    self.stats_R.mean, self.stats_R.mean_sq_sum, self.q_Z.mean,
                    self.q_Z.mean_sq)
        self.q_Z.update(self.X, self.q_pi.ln_mean, self.q_pi.ln_o_mean,
                        self.q_tau.mean, self.q_W.mean, self.q_W.mean_sq,
                        self.stats_R.mean, self.stats_R.mean_sq_sum)
        self.stats_L.update()
        self.q_H.update(self.X, self.q_lambda_H.mean, self.q_tau.mean,
                        self.stats_L.mean, self.stats_L.mean_sq_sum)
        self.stats_R.update()
        # update hyperparameters (under sparse model)
        self.q_lambda_W.update(self.q_W.mean_sq)
        self.q_lambda_H.update(self.q_H.mean)
        self.q_pi.update(self.q_Z.mean)

        # noise update
        self.q_tau.update(self.q_W.mean, self.stats_L.mean_sq_sum,
                          self.q_H.mean, self.stats_R.mean_sq_sum)

    def _ELBO(self):
        base_elbo = self._base_ELBO() # noise elbo is included here
        elbo_W = self.q_W.elbo_part(self.q_lambda_W.mean,
                                    self.q_lambda_W.ln_mean)
        elbo_H = self.q_H.elbo_part(self.q_lambda_H.mean,
                                    self.q_lambda_H.ln_mean)
        elbo_Z = self.q_Z.elbo_part(self.q_pi.ln_mean, self.q_pi.ln_o_mean)
        elbo_lambda_W = self.q_lambda_W.elbo_part()
        elbo_lambda_H = self.q_lambda_H.elbo_part()
        elbo_pi = self.q_pi.elbo_part()
        return base_elbo + elbo_W + elbo_H + elbo_lambda_W + elbo_lambda_H

    def _initialize(self):
        mean_W =  np.empty([self.I, self.K])
        var_W = mean_W.copy()

        mean_H = np.empty([self.K, self.J])
        var_H = mean_H.copy()

        mean_Z = np.empty([self.I,self.K])
        var_Z = mean_Z.copy()

        for i in range(self.I):
            for d in range(self.K):
                estimates  = sample_init_factor_element(
                                                       self.lambda_alpha,
                                                       self.lambda_beta,
                                                       n_samples_lam=10,
                                                       n_samples_F=10)
                mean_W[i,d], var_W[i,d] = estimates


        for d in range(self.K):
            estimates = sample_init_Z_element(self.pi_alpha, self.pi_beta,
                                              self.I,
                                              n_samples_pi=10,
                                              n_samples_Z=10)
            mean_Z[:,d], var_Z[:,d] = estimates

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
        self.q_Z.initialize(mean_Z, var_Z)

        # update other factors once!

        # update hyperparameters (under sparse model)
        self.q_lambda_W.update(self.q_W.mean)
        self.q_lambda_H.update(self.q_H.mean)
        self.q_pi.update(self.q_Z.mean)
        self.stats_L.update()
        self.stats_R.update()
        # noise update
        self.q_tau.update(self.q_W.mean, self.stats_L.mean_sq_sum,
                          self.q_H.mean, self.stats_R.mean_sq_sum)

class GibbsSamplerbnpNMF:

    def __init__(self):
        pass
