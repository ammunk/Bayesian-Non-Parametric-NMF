import numpy as np
from scipy.stats import truncnorm, gamma, expon

from bnsNMF.variational_distributions import WFactor, HFactor,\
                                             FactorPriorGamma,\
                                             FactorPriorGammaARD, NoiseGamma

from bnsNMF.loggers import Logger

class VBBase:

    def __init__(self, X, D, noise_alpha, noise_beta, user_params):
        self.X = X
        self.D = D
        self.alpha_tau = noise_alpha
        self.beta_tau = noise_beta
        self.I, self.J = self.X.shape
        self.max_iter = user_params['max_iter']
        self.tolerance = user_params['tolerance']

        # build model
        self.q_W = WFactor(0, np.infty, self.D)
        self.q_H = HFactor(0, np.infty, self.D)
        self.q_tau = NoiseGamma(self.I*self.J, self.X)

        # calculate various constants
        self.const_likelihood_elbo = -self.I*0.5*np.log(2*np.pi)
        self.X_fro_sq = np.sum(X**2)

    def train(self):

        ELBO_diff = np.inf
        ELBO_old = ELBO_diff
        i = 0
        with Logger(user_params) as logger:
            while ELBO_diff < self.tolerance and i < self.max_iter:
                self._update()

                ELBO = self._ELBO()
                ELBO_diff = ELBO_old - ELBO
                ELBO_old = ELBO
                i = i + 1
                logger.log_update(ELBO, i, self.predict())

    def _update(self):
        pass

    def predict(self):
        return np.dot(self.q_W.mean, self.q_H.mean)

    def transform(self, X):
        pass

    def _ELBO(self):
        pass

    def _base_ELBO(self):
        self.inv_tau_mean, self.ln_tau_mean = self.q_tau.mean,\
                                              self.q_tau.ln_mean

        lik_elbo = self._ELBO_likelihood()
        noise_elbo = self.q_tau.elbo_part(self.alpha_tau, self.beta_tau)

    def _ELBO_likelihood(self):
        W_mean = self.q_W.mean
        W_mean_sq = self.q_W.mean_sq
        H_mean = self.q_H.mean
        H_mean_sq = self.q_H.mean_sq

        XWH = -2*np.sum(self.X * np.dot(W_mean, H_mean))
        mWHsq_sum_sqWH_mean = np.sum(
                            np.dot(W_mean_sq, H_mean_sq)
                            - np.dot(W_mean**2, H_mean**2)
                            )
        ELBO_lik = self.const_likelihood_elbo - self.J*0.5*self.I*ln_tau_mean
        ELBO_lik = ELBO_lik - 0.5*inv_tau_mean * (
                                                  self.X_fro_sq + XWH
                                                  + mWHsq_sum_sqWH_mean
                                                 )
        return ELBO_lik

class psNMF(VBBase):

    def __init__(self, X, D, noise_alpha, noise_beta, 
                 lambda_alpha, lambda_beta, user_params):
        super().__init__(X, D, noise_alpha, noise_beta, user_params)

        self.q_lambda_W = FactorPriorGamma()
        self.q_lambda_H = FactorPriorGamma()

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
        elbo_W = self.q_W.elbo_part(self.q_lambda_W.inv_mean, 
                                    self.q_lambda_W.ln_mean)
        elbo_H = self.q_H.elbo_part(self.q_lambda_H.inv_mean, 
                                    self.q_lambda_H.ln_mean)
        elbo_lambda_W = self.q_lambda_W.elbo_part(self.lambda_alpha, 
                                                  self.lambda_beta)
        elbo_lambda_H = self.q_lambda_H.elbo_part(self.lambda_alpha, 
                                                  self.lambda_beta)

        return base_elbo + elbo_W + elbo_H + elbo_lambda_W + elbo_lambda_H

class pNMF(VBBase):

    def __init__(self, X, D, noise_alpha, noise_beta, 
                 lambda_alpha, lambda_beta):
        super().__init__(X, D, noise_alpha, noise_beta)

        self.q_lambda = FactorPriorGammaARD(self.I + self.J)

    def _update(self):

        # update factors (W and H) one after the other
        self.q_W.update(self.X, self.q_lambda.mean, self.q_tau.mean,
                        self.q_H.mean, self.q_H.mean_sq_sum)
        self.q_H.update(self.X, self.q_lambda.mean, self.q_tau.mean,
                        self.q_W.mean, self.q_W.mean_sq_sum)
        # update hyperparameters (under ARD prior)
        self.q_lambda.update(self.q_W.mean_sq, self.q_H.means_sq)
        # noise update
        self.q_tau.update(self.q_W.mean, self.q_W.mean_sq_sum, 
                          self.q_H.mean, self.q_H.mean_sq_sum)

    def _ELBO(self):
        base_elbo = self._base_ELBO() # noise elbo is included here
        elbo_W = self.q_W.elbo_part(self.q_lambda_W.inv_mean, 
                                    self.q_lambda_W.ln_mean)
        elbo_H = self.q_H.elbo_part(self.q_lambda_H.inv_mean, 
                                    self.q_lambda_H.ln_mean)
        elbo_lambda = self.q_lambda.elbo_part(self.lambda_alpha, 
                                              self.lambda_beta)

        return base_elbo + elbo_W + elbo_H + elbo_lambda_W + elbo_lambda_H
