##############################################################################

# VARIATIONAL INFERENCE / VB

##############################################################################

import sys
import numpy as np

from npbNMF.datatools import mean_X_LR_error_fast
##############################################################################

# Top level base object

##############################################################################

class VBBase:

    def __init__(self, X, K, tolerance, max_iter, a, b):
        from npbNMF.variational_distributions import NoiseGamma
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.noise_alpha = a
        self.noise_beta = b
        self.X = X
        self.K = K
        self.D, self.N = self.X.shape
        self.data_size = self.D*self.N


        self.q_noise = NoiseGamma(self.X, self.noise_alpha, self.noise_beta,
                                  self.data_size)

        self.trained = False

    def train(self, info_dict, log):

        rel_ELBO = np.inf
        ELBO_old = -1e21
        i = 0
        log.info("Initializing")
        info_dict["elbo"] = []
        info_dict["sq_error"] = []
        self._initialize()
        log.info("Finished initializing - beginning training")
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
            info_dict["elbo"].append(ELBO)
            info_dict["sq_error"].append(sum_sq_error)
            msg = (f"\n\tIteration = {i}\n\tELBO = {ELBO}"
                   f"\n\tRelative ELBO = {rel_ELBO}\n"
                   f"\tSq Error = {sum_sq_error}\n============================"
                   "===================")
            log.info(msg)
        self.trained = True

    def _initialize(self):
        pass

    def _update(self):
        pass

    def predict(self):
        pass

    def _ELBO(self):
        pass

    def _ELBO_likelihood(self, mean_L, mean_sq_L, mean_R, mean_sq_R):
        noise_mean, ln_noise_mean = self.q_noise.mean, self.q_noise.ln_mean
        mean_sq_sum_L = np.sum(mean_sq_L, axis=0)
        mean_sq_sum_R = np.sum(mean_sq_R, axis=1)
        sum_mean_sq_error =  mean_X_LR_error_fast(self.X, mean_L,
                                                  mean_sq_sum_L, mean_R,
                                                  mean_sq_sum_R)

        ELBO_lik = self.data_size*0.5*(ln_noise_mean - np.log(2*np.pi))
        ELBO_lik = ELBO_lik - 0.5*noise_mean*(sum_mean_sq_error)
        return ELBO_lik

##############################################################################

# Base object for models without IBP priors

##############################################################################

class BaseBayesNMF(VBBase):

    def __init__(self, X, K, tolerance, max_iter, a, b):
        super().__init__(X, K, tolerance, max_iter, a, b)
        self.q_H = None
        self.q_W = None

    def predict(self):
        return self.q_H.mean @ self.q_W.mean

    def transform(self, X=None):
        if self.trained:
            if X:
                pass
            else:
                return self.q_W.mean
        else:
            sys.exit("ERROR - MODEL NOT TRAINED")

    def extract_features(self):
        if self.trained:
            return self.q_H.mean
        else:
            sys.exit("ERROR - MODEL NOT TRAINED")

    def pickle_factors(self, filename):
        import pickle
        feature_dict = {'H': self.q_H.mean,
                        'W': self.q_W.mean}
        with open(filename, 'bw') as f:
            pickle.dump(feature_dict, f)


class NMFSharedHyperprior(BaseBayesNMF):

    def __init__(self, X, K, tolerance, max_iter, a, b, c, d,
                 prior_type="trunc_norm"):
        super().__init__(X, K, tolerance, max_iter, a, b)
        from npbNMF.variational_distributions import FactorHExpon,\
                                                     FactorHTruncNorm,\
                                                     FactorWExpon,\
                                                     FactorWTruncNorm,\
                                                     HyperpriorExponShared,\
                                                     HyperpriorTruncNormShared
        self.prior_type = prior_type
        self.c = c
        self.d = d
        # build basic model components
        if prior_type == "trunc_norm":
            self.q_W = FactorWTruncNorm(0, np.infty, self.K, self.N)
            self.q_hyperprior = HyperpriorTruncNormShared(c, d, self.D, self.N)
            self.q_H = FactorHTruncNorm(0, np.infty, self.K, self.D)
        elif prior_type == "expon":
            self.q_W = FactorWExpon(0, np.infty, self.K, self.N)
            self.q_hyperprior = HyperpriorExponShared(c, d, self.D, self.N)
            self.q_H = FactorHExpon(0, np.infty, self.K, self.D)
        else:
            sys.exit("Unrecognizable prior for W")

    def _update(self):
        # update factors (W and H) one after the other
        self._update_WH()
        if self.prior_type == "trunc_norm":
            self.q_hyperprior.update(self.q_H.mean_sq, self.q_W.mean_sq)
        elif self.prior_type == "expon":
            self.q_hyperprior.update(self.q_H.mean, self.q_W.mean)
        self._update_noise()

    def _update_WH(self):
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq,
                        self.q_hyperprior.mean, self.q_noise.mean)
        self.q_H.update(self.X, self.q_W.mean, self.q_W.mean_sq,
                        self.q_hyperprior.mean, self.q_noise.mean)

    def _update_noise(self):
        # noise update
        self.q_noise.update(self.q_H.mean, self.q_H.mean_sq,
                            self.q_W.mean, self.q_W.mean_sq)

    def _ELBO(self):
        elbo_lik = self._elbo_lik()
        elbo_noise = self.q_noise.elbo_part()
        elbo_W = self.q_W.elbo_part(self.q_hyperprior.mean,
                                    self.q_hyperprior.ln_mean)
        elbo_H = self.q_H.elbo_part(self.q_hyperprior.mean,
                                    self.q_hyperprior.ln_mean)
        elbo_lambda = self.q_hyperprior.elbo_part()
        return elbo_lik + elbo_W + elbo_H + elbo_lambda + elbo_noise

    def _elbo_lik(self):
        elbo_lik = self._ELBO_likelihood(self.q_H.mean, self.q_H.mean_sq,
                                         self.q_W.mean, self.q_W.mean_sq)
        return elbo_lik

    def _initialize(self):

        expected_hyperprior = self.c/self.d
        self.q_W.initialize(expected_hyperprior)
        self.q_H.initialize(expected_hyperprior)
        # update other parts once
        if self.prior_type == "trunc_norm":
            self.q_hyperprior.update(self.q_H.mean_sq, self.q_W.mean_sq)
        elif self.prior_type == "expon":
            self.q_hyperprior.update(self.q_H.mean, self.q_W.mean)
        else:
            sys.exit("Unkown hyperprior - in initialization")
        self._update_noise()

class NMFDoubleHyperprior(BaseBayesNMF):

    def __init__(self, X, K, tolerance, max_iter, a, b, c, d, e, f,
                 prior_type="trunc_norm"):
        super().__init__(X, K, tolerance, max_iter, a, b)
        from npbNMF.variational_distributions import FactorHExpon,\
                                                     FactorHTruncNorm,\
                                                     FactorWExpon,\
                                                     FactorWTruncNorm,\
                                                     HyperpriorHExpon,\
                                                     HyperpriorHTruncNorm,\
                                             HyperpriorTruncNormWSparsity,\
                                             HyperpriorExponWSparsity
        self.prior_type = prior_type
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        # build basic model components
        if prior_type == "trunc_norm":
            self.q_W = FactorWTruncNorm(0, np.infty, self.K, self.N)
            self.q_hyperprior_W = HyperpriorTruncNormWSparsity(e, f)
            self.q_H = FactorHTruncNorm(0, np.infty, self.K, self.D)
            self.q_hyperprior_H = HyperpriorHTruncNorm(c, d, self.D)
        elif prior_type == "expon":
            self.q_W = FactorWExpon(0, np.infty, self.K, self.N)
            self.q_hyperprior_W = HyperpriorExponWSparsity(e, f)
            self.q_H = FactorHExpon(0, np.infty, self.K, self.D)
            self.q_hyperprior_H = HyperpriorHExpon(c, d, self.D)
        else:
            sys.exit("Unrecognizable prior for W")

    def _update(self):
        # update factors (W and H) one after the other
        self._update_WH()
        # update hyperparameters (under sparse model)
        if self.prior_type == "trunc_norm":
            self.q_hyperprior_W.update(self.q_W.mean_sq)
            self.q_hyperprior_H.update(self.q_H.mean_sq)
        elif self.prior_type == "expon":
            self.q_hyperprior_W.update(self.q_W.mean)
            self.q_hyperprior_H.update(self.q_H.mean)

        self._update_noise()

    def _update_WH(self):
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq,
                        self.q_hyperprior_W.mean, self.q_noise.mean)
        self.q_H.update(self.X, self.q_W.mean, self.q_W.mean_sq,
                        self.q_hyperprior_H.mean, self.q_noise.mean)

    def _update_noise(self):
        # noise update
        self.q_noise.update(self.q_H.mean, self.q_H.mean_sq,
                            self.q_W.mean, self.q_W.mean_sq)

    def _ELBO(self):
        elbo_lik = self._elbo_lik()
        elbo_noise = self.q_noise.elbo_part()
        elbo_W = self.q_W.elbo_part(self.q_hyperprior_W.mean,
                                    self.q_hyperprior_W.ln_mean)
        elbo_H = self.q_H.elbo_part(self.q_hyperprior_H.mean,
                                    self.q_hyperprior_H.ln_mean)
        elbo_hyperprior_W = self.q_hyperprior_W.elbo_part()
        elbo_hyperprior_H = self.q_hyperprior_H.elbo_part()
        return elbo_lik + elbo_W + elbo_H + elbo_hyperprior_W\
               + elbo_hyperprior_H + elbo_noise

    def _elbo_lik(self):
        elbo_lik = self._ELBO_likelihood(self.q_H.mean, self.q_H.mean_sq,
                                         self.q_W.mean, self.q_W.mean_sq)
        return elbo_lik

    def _initialize(self):

        expected_hyperprior_H = self.c/self.d
        expected_hyperprior_W = self.e/self.f
        self.q_W.initialize(expected_hyperprior_W)
        self.q_H.initialize(expected_hyperprior_H)
        # update other factors once!
        if self.prior_type == "trunc_norm":
            self.q_hyperprior_W.update(self.q_W.mean_sq)
            self.q_hyperprior_H.update(self.q_H.mean_sq)
        elif self.prior_type == "expon":
            self.q_hyperprior_W.update(self.q_W.mean)
            self.q_hyperprior_H.update(self.q_H.mean)
        else:
            sys.exit("Unkown hyperprior - in initialization")
        self._update_noise()

class NMFDoubleHyperpriorWithShared(BaseBayesNMF):

    def __init__(self, X, K, tolerance, max_iter, a, b, c, d, e, f,
                 prior_type="trunc_norm"):
        super().__init__(X, K, tolerance, max_iter, a, b)
        from npbNMF.variational_distributions import FactorHExpon,\
                                                     FactorHTruncNorm,\
                                                     FactorWExpon,\
                                                     FactorWTruncNorm,\
                                       HyperpriorTruncNormSharedWithSparsity,\
                                       HyperpriorExponSharedWithSparsity,\
                                       HyperpriorTruncNormSparsityWithShared,\
                                       HyperpriorExponSparsityWithShared
        self.prior_type = prior_type
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        # build basic model components
        if prior_type == "trunc_norm":
            self.q_W = FactorWTruncNorm(0, np.infty, self.K, self.N)
            self.q_H = FactorHTruncNorm(0, np.infty, self.K, self.D)
            self.q_hyperprior_W= HyperpriorTruncNormSparsityWithShared(e, f,
                                                                       self.N,
                                                                       self.K)
            self.q_hyperprior_shared = HyperpriorTruncNormSharedWithSparsity(c,
                                                                             d,
                                                                        self.D,
                                                                        self.N,
                                                                        self.K)

        elif prior_type == "expon":
            self.q_W = FactorWExpon(0, np.infty, self.K, self.N)
            self.q_H = FactorHExpon(0, np.infty, self.K, self.D)
            self.q_hyperprior_W= HyperpriorExponSparsityWithShared(e, f,
                                                                    self.N,
                                                                    self.K)
            self.q_hyperprior_shared = HyperpriorExponSharedWithSparsity(c,
                                                                         d,
                                                                        self.D,
                                                                        self.N,
                                                                        self.K)
        else:
            sys.exit("Unrecognizable prior for W and H")

    def _update(self):
        # update factors (W and H) one after the other
        self._update_WH()
        # update hyperparameters (under sparse model)
        if self.prior_type == "trunc_norm":
            self.q_hyperprior_shared.update(self.q_H.mean_sq, self.q_W.mean_sq,
                                            self.q_hyperprior_W.mean)
            mean_hyp_prior = self.q_hyperprior_shared.mean
            self.q_hyperprior_W.update(self.q_W.mean_sq, mean_hyp_prior)
        elif self.prior_type == "expon":
            self.q_hyperprior_shared.update(self.q_H.mean, self.q_W.mean,
                                            self.q_hyperprior_W.mean)
            mean_hyp_prior = self.q_hyperprior_shared.mean
            self.q_hyperprior_W.update(self.q_W.mean, mean_hyp_prior)
        self._update_noise()

    def _update_WH(self):
        hyp_prod = self.q_hyperprior_W.mean.T*self.q_hyperprior_shared.mean
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq, hyp_prod.T,
                        self.q_noise.mean)
        self.q_H.update(self.X, self.q_W.mean, self.q_W.mean_sq,
                        self.q_hyperprior_shared.mean, self.q_noise.mean)

    def _update_noise(self):
        # noise update
        self.q_noise.update(self.q_H.mean, self.q_H.mean_sq,
                            self.q_W.mean, self.q_W.mean_sq)

    def _ELBO(self):
        elbo_lik = self._elbo_lik()
        elbo_noise = self.q_noise.elbo_part()
        mean_hyp_prod = self.q_hyperprior_W.mean.T\
                        *self.q_hyperprior_shared.mean
        ln_mean_hyp_sum = self.q_hyperprior_W.ln_mean.T\
                          + self.q_hyperprior_shared.ln_mean
        elbo_W = self.q_W.elbo_part(mean_hyp_prod.T, ln_mean_hyp_sum.T)
        elbo_H = self.q_H.elbo_part(self.q_hyperprior_shared.mean,
                                    self.q_hyperprior_shared.ln_mean)
        elbo_hyperprior_W = self.q_hyperprior_W.elbo_part()
        elbo_hyperprior_H = self.q_hyperprior_shared.elbo_part()
        return elbo_lik + elbo_W + elbo_H + elbo_hyperprior_W\
               + elbo_hyperprior_H + elbo_noise

    def _elbo_lik(self):
        elbo_lik = self._ELBO_likelihood(self.q_H.mean, self.q_H.mean_sq,
                                         self.q_W.mean, self.q_W.mean_sq)
        return elbo_lik

    def _initialize(self):

        expected_hyperprior_shared = self.c/self.d
        expected_hyperprior_W = self.e/self.f
        self.q_W.initialize(expected_hyperprior_W*expected_hyperprior_shared)
        self.q_H.initialize(expected_hyperprior_shared)
        # update other factors once!
        if self.prior_type == "trunc_norm":
            self.q_hyperprior_shared.update(self.q_H.mean_sq, self.q_W.mean_sq,
                                            self.q_hyperprior_W.mean)
            mean_hyp_prior = self.q_hyperprior_shared.mean
            self.q_hyperprior_W.update(self.q_W.mean_sq, mean_hyp_prior)
        elif self.prior_type == "expon":
            self.q_hyperprior_shared.update(self.q_H.mean, self.q_W.mean,
                                            self.q_hyperprior_W.mean)
            mean_hyp_prior = self.q_hyperprior_shared.mean
            self.q_hyperprior_W.update(self.q_W.mean, mean_hyp_prior)
        else:
            sys.exit("Unkown hyperprior - in initialization")
        self._update_noise()

##############################################################################

# Base object for models with IBP priors

##############################################################################

class BaseNPB:

    def __init__(self, X, K, pi_alpha, pi_beta, q_H, trained,
                 prior_type="trunc_norm"):
        from npbNMF.variational_distributions import FactorBernoulli,\
                                                     NPBFactorWExpon,\
                                                     NPBFactorWTruncNorm,\
                                                     BetaPrior
        D, N = X.shape
        self.X = X
        self.pi_alpha = pi_alpha
        self.pi_beta = pi_beta
        self.q_H = q_H
        self.trained = trained
        if prior_type == "trunc_norm":
            self.q_W = NPBFactorWTruncNorm(0, np.infty, K, N)
        elif prior_type == "expon":
            self.q_W = NPBFactorWExpon(0, np.infty, K, N)
        else:
            sys.exit("Unrecognizable prior for W")
        self.q_Z = FactorBernoulli(K, N)
        self.q_pi = BetaPrior(N, K, pi_alpha, pi_beta)

    def _update(self, H_mean, H_mean_sq, W_mean, W_mean_sq, noise_mean):
        self.q_Z.update(self.X, H_mean, H_mean_sq, W_mean, W_mean_sq,
                        self.q_pi.ln_mean, self.q_pi.ln_o_mean, noise_mean)
        self.q_pi.update(self.q_Z.mean)

    def _ELBO(self):
        elbo_Z = self.q_Z.elbo_part(self.q_pi.ln_mean, self.q_pi.ln_o_mean)
        elbo_pi = self.q_pi.elbo_part()
        return elbo_Z + elbo_pi

    def _initialize(self):
        self.q_Z.initialize(self.pi_alpha, self.pi_beta)
        self.q_pi.update(self.q_Z.mean)

    def predict(self):
        return self.q_H.mean @ (self.q_W.mean*self.q_Z.mean)

    def transform(self, X=None):
        if self.trained:
            if X:
                pass
            else:
                return self.q_Z.mean*self.q_W.mean
        else:
            sys.exit("ERROR - MODEL NOT TRAINED")

    def pickle_factors(self, filename):
        import pickle
        feature_dict = {'H': self.q_H.mean,
                        'W': self.q_W.mean,
                        'Z': self.q_Z.mean}
        with open(filename, 'bw') as f:
            pickle.dump(feature_dict, f)

class NPBNMFSharedHyperprior(BaseNPB, NMFSharedHyperprior): # MUST BE THIS
                                                            # ORDER

    def __init__(self, X, K, tolerance, max_iter, a, b, c, d, pi_alpha,
                 pi_beta, prior_type="trunc_norm"):
        NMFSharedHyperprior.__init__(self, X, K, tolerance, max_iter, a, b, c,
                                     d, prior_type=prior_type)
        BaseNPB.__init__(self, X, K, pi_alpha, pi_beta, self.q_H, self.trained,
                         prior_type)

    def _update(self):
        BaseNPB._update(self, self.q_H.mean, self.q_H.mean_sq, self.q_W.mean,
                        self.q_W.mean_sq, self.q_noise.mean)
        NMFSharedHyperprior._update(self)

    def _update_WH(self):
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq, self.q_Z.mean,
                        self.q_Z.mean_sq, self.q_hyperprior.mean,
                        self.q_noise.mean)
        self.q_H.update(self.X, self.q_W.mean*self.q_Z.mean,
                        self.q_W.mean_sq*self.q_Z.mean_sq,
                        self.q_hyperprior.mean, self.q_noise.mean)

    def _update_noise(self):
        # noise update
        self.q_noise.update(self.q_H.mean, self.q_H.mean_sq,
                            self.q_W.mean*self.q_Z.mean,
                            self.q_W.mean_sq*self.q_Z.mean_sq)

    def _ELBO(self):
        elbo_IBP = BaseNPB._ELBO(self)
        elbo_rest = NMFSharedHyperprior._ELBO(self)
        return elbo_IBP + elbo_rest

    def _elbo_lik(self):
        elbo_lik = self._ELBO_likelihood(self.q_H.mean, self.q_H.mean_sq,
                                         self.q_W.mean*self.q_Z.mean,
                                         self.q_W.mean_sq*self.q_Z.mean_sq)
        return elbo_lik

    def _initialize(self):
        BaseNPB._initialize(self)
        NMFSharedHyperprior._initialize(self)

class NPBNMFDoubleHyperprior(BaseNPB, NMFDoubleHyperprior):

    def __init__(self, X, K, tolerance, max_iter, a, b, c, d, e, f,
                 pi_alpha, pi_beta, prior_type="trunc_norm"):
        NMFDoubleHyperprior.__init__(self, X, K, tolerance, max_iter, a, b, c,
                                     d, e, f, prior_type=prior_type)
        BaseNPB.__init__(self, X, K, pi_alpha, pi_beta, self.q_H, self.trained,
                         prior_type)

    def _update(self):
        BaseNPB._update(self, self.q_H.mean, self.q_H.mean_sq, self.q_W.mean,
                        self.q_W.mean_sq, self.q_noise.mean)
        NMFDoubleHyperprior._update(self)

    def _update_WH(self):
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq, self.q_Z.mean,
                        self.q_Z.mean_sq, self.q_hyperprior_W.mean,
                        self.q_noise.mean)
        self.q_H.update(self.X, self.q_W.mean*self.q_Z.mean,
                        self.q_W.mean_sq*self.q_Z.mean_sq,
                        self.q_hyperprior_H.mean,
                        self.q_noise.mean)

    def _update_noise(self):
        # noise update
        self.q_noise.update(self.q_H.mean, self.q_H.mean_sq,
                            self.q_W.mean*self.q_Z.mean,
                            self.q_W.mean_sq*self.q_Z.mean_sq)

    def _ELBO(self):
        elbo_IBP = BaseNPB._ELBO(self)
        elbo_rest = NMFDoubleHyperprior._ELBO(self)
        return elbo_IBP + elbo_rest

    def _elbo_lik(self):
        elbo_lik = self._ELBO_likelihood(self.q_H.mean, self.q_H.mean_sq,
                                         self.q_W.mean*self.q_Z.mean,
                                         self.q_W.mean_sq*self.q_Z.mean_sq)
        return elbo_lik

    def _initialize(self):
        BaseNPB._initialize(self)
        NMFDoubleHyperprior._initialize(self)

class NPBNMFDoubleHyperpriorWithShared(BaseNPB, NMFDoubleHyperpriorWithShared):

    def __init__(self, X, K, tolerance, max_iter, a, b, c, d, e, f,
                 pi_alpha, pi_beta, prior_type="trunc_norm"):
        NMFDoubleHyperpriorWithShared.__init__(self, X, K, tolerance, max_iter,
                                               a, b, c, d, e, f,
                                               prior_type=prior_type)
        BaseNPB.__init__(self, X, K, pi_alpha, pi_beta, self.q_H, self.trained,
                         prior_type)

    def _update(self):
        BaseNPB._update(self, self.q_H.mean, self.q_H.mean_sq, self.q_W.mean,
                        self.q_W.mean_sq, self.q_noise.mean)
        NMFDoubleHyperpriorWithShared._update(self)

    def _update_WH(self):
        prod = self.q_hyperprior_W.mean.T*self.q_hyperprior_shared.mean
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq, self.q_Z.mean,
                        self.q_Z.mean_sq, prod.T, self.q_noise.mean)
        self.q_H.update(self.X, self.q_W.mean*self.q_Z.mean,
                        self.q_W.mean_sq*self.q_Z.mean_sq,
                        self.q_hyperprior_shared.mean, self.q_noise.mean)

    def _update_noise(self):
        # noise update
        self.q_noise.update(self.q_H.mean, self.q_H.mean_sq,
                            self.q_W.mean*self.q_Z.mean,
                            self.q_W.mean_sq*self.q_Z.mean_sq)

    def _ELBO(self):
        elbo_IBP = BaseNPB._ELBO(self)
        elbo_rest = NMFDoubleHyperpriorWithShared._ELBO(self)
        return elbo_IBP + elbo_rest

    def _elbo_lik(self):
        elbo_lik = self._ELBO_likelihood(self.q_H.mean, self.q_H.mean_sq,
                                         self.q_W.mean*self.q_Z.mean,
                                         self.q_W.mean_sq*self.q_Z.mean_sq)
        return elbo_lik

    def _initialize(self):
        BaseNPB._initialize(self)
        NMFDoubleHyperpriorWithShared._initialize(self)
