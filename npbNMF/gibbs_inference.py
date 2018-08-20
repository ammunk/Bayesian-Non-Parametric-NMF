##############################################################################

# MCMC INFERENCE
# USING Gibbs & Metropolis-Hastings

##############################################################################

from random import shuffle
import numpy as np
from scipy.stats import poisson, expon
from npbNMF.datatools import truncated_sampler, trunc_norm_sum_log_pdf, \
                             gamma_sum_log_pdf, IBP_sum_log_pdf

##############################################################################

# Base Gibbs

##############################################################################

class BaseGibbsSampler:

    def __init__(self, X, K_init, a, b, g, l, m, o, num_samples=2000,
                 alpha_beta_prior=False):
        from npbNMF.gibbs_distributions import FactorW, FactorH, FactorZ,\
                                               NoisePrior, IBPAlphaPrior,\
                                               IBPBetaPrior
        self.a = a
        self.b = b
        self.X = X
        self.K_init = K_init
        self.D, self.N = X.shape
        self.alpha_beta_prior = alpha_beta_prior
        self.num_samples = num_samples
        self.trunc_low_bound = 0
        self.trunc_up_bound = np.infty

        self.pZ = FactorZ(self.D, self.N, K_init)
        self.pW = FactorW(0, np.infty, self.N, K_init)
        self.pH = FactorH(0, np.infty, self.D, K_init)
        self.pNoise = NoisePrior(self.D*self.N, X, a, b)
        if alpha_beta_prior == False:
            self.alpha = g/l
            self.beta = m/o
        else:
            self.IBPalpha = IBPAlphaPrior(g, l, self.N)
            self.IBPbeta = IBPBetaPrior(m, o, self.N)

        self.samples = {"W": [],
                        "H": [],
                        "Z": [],
                        "noise": [],
                        "active_features": [],
                        "M": []}

        if alpha_beta_prior:
            self.samples["IBPAlpha"] = []
            self.samples["IBPBeta"] = []
        else:
            pass

    def _initialize(self):
        if self.alpha_beta_prior:
            self.IBPalpha.initialize()
            self.IBPbeta.initialize()
        self.pZ.initialize()
        self.pNoise.initialize()
        self._initialize_factor_hyperprior()

    def _initialize_factor_hyperprior(self):
        pass

    def train(self, info_dict, log):

        i = 0
        log.info("Initializing")
        self._initialize()
        log.info("Finished initializing - beginning training")
        info_dict["log_joint"] = []
        info_dict["K_feature_change"] = []
        info_dict["K_plus"] = []
        info_dict["weighted_K_plus"] = []
        info_dict["n_features"] = []
        info_dict["weighted_n_features"] = []
        K_plus_old = self.K_init

        while i < self.num_samples:
            # sample is a dictionary containing one sample set to be stored
            samples = self._sample()
            self._store_samples(samples)
            log_joint_posterior = self._log_joint()
            info_dict["log_joint"].append(log_joint_posterior)
            if self.pZ.K_plus != K_plus_old:
                info_dict["K_feature_change"].append(i)
                K_plus_old = self.pZ.K_plus
            info_dict["K_plus"].append(self.pZ.K_plus)
            info_dict["weighted_K_plus"].append(np.sum(np.sum(self.pZ.samples
                                                       *self.pW.samples,
                                                       axis=1)>1e-16))
            info_dict["n_features"].append(np.sum(self.pZ.samples,
                                                  axis=0).tolist())
            info_dict["weighted_n_features"].append(np.sum(self.pZ.samples
                                                           *self.pW.samples
                                                           >1e-16,
                                                           axis=0).tolist())
            msg = (f"\n\tIteration = {i}")
            msg += (f"\n\t- Active features = {self.pZ.K_plus}")
            msg += "\n===============================================\n"
            log.info(msg)
            i += 1
        info_dict["sq_error"] = np.sum((self.X-self.predict())**2)


    def _sample(self):

        self._sample_WH()
        self._sample_Z()

        self.prune_inactive_features()

        if self.alpha_beta_prior:
            self.IBPalpha.update_and_sample(self.IBPbeta.samples,
                                            self.pZ.K_plus)
            self.IBPbeta.update_and_sample(self.IBPalpha.samples,
                                           self.pZ.K_plus, self.pZ.M)

        self.pNoise.update_and_sample(self.pH.samples,
                                      self.pW.samples*self.pZ.samples)
        hyper_prior_results = self._sample_hyperpriors()

        result ={"W": self.pW.samples.copy(),
                 "H": self.pH.samples.copy(),
                 "Z": self.pZ.samples.copy(),
                 "noise": self.pNoise.samples.copy(),
                 "active_features": self.pZ.K_plus, # integer, thus no copy
                 "M": self.pZ.M.copy()}

        for key, item in hyper_prior_results.items():
            result[key] = item

        if self.alpha_beta_prior:
                 result["IBPAlpha"] = self.IBPalpha.samples.copy()
                 result["IBPBeta"] = self.IBPbeta.samples.copy()

        return result

    def _sample_Z(self):
        if self.alpha_beta_prior:
            alpha = self.IBPalpha.samples
            beta = self.IBPbeta.samples
        else:
            alpha = self.alpha
            beta = self.beta

        for n in np.random.permutation(self.N):
            Xn = self.X[:,n]
            Wn = self.pW.samples[:,n]
            self.pZ.update_and_sample_zn(Xn,
                                         self.pH.samples,
                                         Wn,
                                         alpha,
                                         beta,
                                         self.pNoise.samples,
                                         n)
            Zn = self.pZ.samples[:,n]
            En = Xn.reshape(-1,1) - self.pH.samples @ (Wn*Zn).reshape(-1,1)
            tmp = self._sample_Kn(En, alpha, beta, self.pNoise.samples)
            is_new, k_new, new = tmp
            if is_new:
                new_H, new_Wn, new_hyperpriors= new
                self.pH.add_new_features(new_H, k_new)
                self.pZ.add_Zn_samples(k_new, n)
                self._add_new_W_hyperpriors(new_Wn, new_hyperpriors, k_new, n)

    def _add_new_W_hyperpriors(self, new_Wn, new_hyperpriors, k_new, n):
        pass

    def _sample_Kn(self, En, alpha, beta, noise):
        """ sampling new features using MH-sampling

            This is done for a single observation only, thus all active samples
            and hyperpriors relates to only a single observation n
        """
        poisson_parameter = (alpha*beta)/(beta + self.N - 1)
        n_new_features = poisson.rvs(poisson_parameter)
        accepted = False
        if n_new_features == 0:
            # In this case no features are sampled and there were no new
            # features in last iteration - thus this proposal is always
            # accepted - equivalent to rejected.
            # This check is a special case, and theoretically
            # redundant, but happens quite often, thus computational time
            # can be reduced.
            accepted = False
        else:
            new_params = self._new_params(n_new_features)
            new_H, new_Wn, new_hyperpriors = new_params
            newHnewW = (new_H @ new_Wn.reshape(-1,1))
            diff_new = newHnewW - 2*En
            exponent = newHnewW.T @ diff_new
            logr = -0.5*noise*(exponent)

            if logr > 0 or np.random.uniform() < np.exp(logr):
                accepted_H = new_H
                accepted_Wn = new_Wn
                accepted_hyperpriors = new_hyperpriors
                accepted = True
            else:
                accepted = False

        if accepted:
            # sample the last element of the markov chain, which is least
            # correlated wih the initial value
            return True, n_new_features, (accepted_H.copy(),
                                          accepted_Wn.copy(),
                                          accepted_hyperpriors)
        else:
            return False, None, None

    def _new_params(self, n_new_features):
        pass

    def _sample_WH(self):
        pass

    def _sample_hyperpriors(self):
        pass

    def prune_inactive_features(self):
        """
        This function prunes features which has become inactive.
            - Frees memory

        """
        self.pZ.prune_features()
        mask, K_plus = self.pZ.active_mask, self.pZ.K_plus
        self.pW.prune_features(mask, K_plus)
        self.pH.prune_features(mask, K_plus)
        self._prune_hyperpriors(mask, K_plus)

    def _prune_hyperpriors(self, mask, K_plus):
        pass

    def predict(self):
        burnin = 200
        H, W, Z = self.samples["H"], self.samples["W"], self.samples["Z"]
        pred = np.zeros([self.D, self.N, len(H)-burnin])
        k  = 0
        for i, (h,w,z) in enumerate(zip(H,W,Z)):
            if i >= burnin:
                pred[:,:,k] = h @ (w*z)
                k += 1

        return pred.mean(axis=2)

    def extract_features(self):
        features = self.samples["H"][-1]
        mask = self.samples["M"][-1] > 0
        return features[:,mask]

    def transform(self):
        WZ = self.samples["W"][-1]*self.samples["Z"][-1]
        return WZ

    def _store_samples(self, samples):
        for key, sample in samples.items():
            self.samples[key].append(sample)

    def _log_joint(self):
        pass

    def _log_likelihood(self, H, WZ, noise):
        return 0.5*((self.N*self.D)*(-np.log(2*np.pi) + np.log(noise))
                    - noise*((self.X - H@WZ).ravel()@(self.X - H@WZ).ravel()))

    def _log_noise(self, samples):
        return gamma_sum_log_pdf(samples, self.a, self.b)

    def pickle_factors(self, filename):
        import pickle
        feature_dict = {'H': self.pH.samples,
                        'W': self.pW.samples,
                        'Z': self.pZ.samples}
        with open(filename, 'bw') as f:
            pickle.dump(feature_dict, f)

##############################################################################

# Inference models

##############################################################################

class GibbsSharedNPB(BaseGibbsSampler):

    def __init__(self, X, K_init, a, b, c, d, g, l, m, o, num_samples=2000,
                 alpha_beta_prior=False):
        super().__init__(X, K_init, a, b, g, l, m, o, num_samples=num_samples,
                         alpha_beta_prior=alpha_beta_prior)
        from npbNMF.gibbs_distributions import HyperpriorShared

        self.c = c
        self.d = d
        self.pSharedHyp= HyperpriorShared(c, d, self.D, self.N, K_init)
        self.samples["hyp_shared"] = []

    def _initialize_factor_hyperprior(self):
        self.pSharedHyp.initialize()
        self.pW.initialize(self.pSharedHyp.samples)
        self.pH.initialize(self.pSharedHyp.samples)

    def _new_params(self, n_new_features):
        new_H = np.zeros([self.D, n_new_features])
        new_Wn = np.zeros(n_new_features)
        shared_hyperprior = np.zeros(n_new_features)

        for k in range(n_new_features):
            hyp_prior = self.pSharedHyp.prior.rvs()
            # loop through the rows and assign samples
            new_Wn[k] = truncated_sampler(mu=0,
                                          std=1/np.sqrt(hyp_prior),
                                          lower_bound=self.trunc_low_bound,
                                          upper_bound=self.trunc_up_bound)
            shared_hyperprior[k] = hyp_prior
            for d in range(self.D):
                new_H[d,k] =  truncated_sampler(mu=0,
                                              std=1/np.sqrt(hyp_prior),
                                              lower_bound=self.trunc_low_bound,
                                              upper_bound=self.trunc_up_bound)
        return new_H, new_Wn, shared_hyperprior

    def _add_new_W_hyperpriors(self, new_Wn, new_hyperpriors, k_new, n):
        self.pSharedHyp.add_new_features(new_hyperpriors, k_new)
        add_W = np.empty((k_new, self.N))
        for i, param in enumerate(new_hyperpriors):
            param = np.broadcast_to(param, self.N)
            add_W[i,:] = truncated_sampler(mu=0,
                                           std=1/np.sqrt(param),
                                           lower_bound=self.trunc_low_bound,
                                           upper_bound=self.trunc_up_bound)
        add_W[:,n] = new_Wn
        self.pW.add_new_features(add_W, k_new, n)

    def _sample_WH(self):
        self.pW.update_and_sample(self.X, self.pH.samples, self.pZ.samples,
                                  self.pSharedHyp.samples,
                                  self.pNoise.samples)
        self.pH.update_and_sample(self.X, self.pW.samples*self.pZ.samples,
                                  self.pSharedHyp.samples,
                                  self.pNoise.samples)

    def _sample_hyperpriors(self):
        self.pSharedHyp.update_and_sample(self.pH.samples, self.pW.samples)
        result = {"hyp_shared": self.pSharedHyp.samples}
        return result

    def _prune_hyperpriors(self, mask, K_plus):
        self.pSharedHyp.prune_features(mask, K_plus)

    def _log_joint(self):
        log_likelihood = self._log_likelihood(self.pH.samples,
                                              self.pW.samples*self.pZ.samples,
                                              self.pNoise.samples)
        #log_prior = self._log_prior(self.pSharedHyp.samples)
        #log_hyper_prior = self._log_hyperprior()
        #log_noise = self._log_noise(self.pNoise.samples)
        #log_IBP = self._log_IBP()
        return log_likelihood #+ log_prior + log_hyper_prior + log_IBP

    def _log_prior(self, hyperprior_shared):
        logpdf = trunc_norm_sum_log_pdf(self.pW.samples.T, hyperprior_shared)
        logpdf += trunc_norm_sum_log_pdf(self.pH.samples, hyperprior_shared)

        return np.sum(logpdf)

    def _log_hyperprior(self):
        logpdf = gamma_sum_log_pdf(self.pSharedHyp.samples, self.c, self.d)
        return np.sum(logpdf)

    def _log_IBP(self):
        log_pdf = IBP_sum_log_pdf(self.pZ.samples, self.alpha, self.beta,
                                  self.pZ.K_plus, self.pZ.M, self.N)
        return log_pdf

class GibbsSparseNPB(BaseGibbsSampler):

    def __init__(self, X, K_init, a, b, c, d, e, f, g, l, m, o,
                 num_samples=2000, alpha_beta_prior=False):
        super().__init__(X, K_init, a, b, g, l, m, o, num_samples=num_samples,
                         alpha_beta_prior=alpha_beta_prior)
        from npbNMF.gibbs_distributions import HyperpriorH, HyperpriorSparse

        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.pHypH= HyperpriorH(c, d, self.D, K_init)
        self.pHypW= HyperpriorSparse(e, f, self.N, K_init)
        self.samples["hyp_H"] = []
        self.samples["hyp_W"] = []

    def _initialize_factor_hyperprior(self):
        self.pHypH.initialize()
        self.pHypW.initialize()
        self.pW.initialize(self.pHypW.samples)
        self.pH.initialize(self.pHypH.samples)

    def _new_params(self, n_new_features):
        new_H = np.zeros([self.D, n_new_features])
        new_Wn = np.zeros(n_new_features)
        hyperprior_H = np.zeros(n_new_features)
        hyperprior_W = np.zeros(n_new_features)

        for k in range(n_new_features):
            hyp_prior = self.pHypW.prior.rvs()
            # loop through the rows and assign samples
            new_Wn[k] = truncated_sampler(mu=0,
                                          std=1/np.sqrt(hyp_prior),
                                          lower_bound=self.trunc_low_bound,
                                          upper_bound=self.trunc_up_bound)
            hyperprior_W[k] = hyp_prior
            hyp_prior = self.pHypH.prior.rvs()
            hyperprior_H[k] = hyp_prior
            for d in range(self.D):
                new_H[d,k] = truncated_sampler(mu=0,
                                              std=1/np.sqrt(hyp_prior),
                                              lower_bound=self.trunc_low_bound,
                                              upper_bound=self.trunc_up_bound)


        return new_H, new_Wn, (hyperprior_H, hyperprior_W)

    def _add_new_W_hyperpriors(self, new_Wn, new_hyperpriors, k_new, n):
        new_hyp_H, new_hyp_Wn = new_hyperpriors
        self.pHypH.add_new_features(new_hyp_H, k_new)
        add_W = np.empty((k_new, self.N))
        add_hyp_W = add_W.copy()
        for n in range(self.N):
            for k in range(k_new):
                hyp_prior = self.pHypW.prior.rvs()
                add_hyp_W[k,n] = hyp_prior
                add_W[k,n] = truncated_sampler(mu=0,
                                              std=1/np.sqrt(hyp_prior),
                                              lower_bound=self.trunc_low_bound,
                                              upper_bound=self.trunc_up_bound)
        add_hyp_W[:,n] = new_hyp_Wn
        add_W[:,n] = new_Wn
        self.pW.add_new_features(add_W, k_new, n)
        self.pHypW.add_new_features(add_hyp_W, k_new)

    def _sample_WH(self):
        self.pW.update_and_sample(self.X, self.pH.samples, self.pZ.samples,
                                  self.pHypW.samples, self.pNoise.samples)
        self.pH.update_and_sample(self.X, self.pW.samples*self.pZ.samples,
                                  self.pHypH.samples, self.pNoise.samples)

    def _sample_hyperpriors(self):
        self.pHypH.update_and_sample(self.pH.samples)
        self.pHypW.update_and_sample(self.pW.samples)
        result = {"hyp_H": self.pHypH.samples,
                  "hyp_W": self.pHypW.samples}
        return result

    def _prune_hyperpriors(self, mask, K_plus):
        self.pHypH.prune_features(mask, K_plus)
        self.pHypW.prune_features(mask, K_plus)

    def _log_joint(self):
        log_likelihood = self._log_likelihood(self.pH.samples,
                                              self.pW.samples*self.pZ.samples,
                                              self.pNoise.samples)
        #log_prior = self._log_prior(self.pHypH.samples, self.pHypW.samples)
        #log_hyper_prior = self._log_hyperprior()
        #log_noise = self._log_noise(self.pNoise.samples)
        #log_IBP = self._log_IBP()
        return log_likelihood #+ log_prior + log_hyper_prior + log_IBP

    def _log_prior(self, hyperprior_H, hyperprior_W):
        logpdf = trunc_norm_sum_log_pdf(self.pW.samples, hyperprior_W)
        logpdf += trunc_norm_sum_log_pdf(self.pH.samples, hyperprior_H)

        return np.sum(logpdf)

    def _log_hyperprior(self):
        logpdf = gamma_sum_log_pdf(self.pHypH.samples, self.c, self.d)
        logpdf += gamma_sum_log_pdf(self.pHypW.samples, self.e, self.f)
        return np.sum(logpdf)

    def _log_IBP(self):
        log_pdf = IBP_sum_log_pdf(self.pZ.samples, self.alpha, self.beta,
                                  self.pZ.K_plus, self.pZ.M, self.N)
        return log_pdf

class GibbsSharedSparseNPB(BaseGibbsSampler):

    def __init__(self, X, K_init, a, b, c, d, e, f, g, l, m, o,
                 num_samples=2000, alpha_beta_prior=False):
        super().__init__(X, K_init, a, b, g, l, m, o, num_samples=num_samples,
                         alpha_beta_prior=alpha_beta_prior)
        from npbNMF.gibbs_distributions import HyperpriorSharedWithSparsity,\
                                               HyperpriorSparseWithShared

        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.pSharedHyp= HyperpriorSharedWithSparsity(c, d, self.D, self.N,
                                                      K_init)
        self.pHypW= HyperpriorSparseWithShared(e, f, self.N, K_init)
        self.samples["hyp_shared"] = []
        self.samples["hyp_W"] = []

    def _initialize_factor_hyperprior(self):
        self.pSharedHyp.initialize()
        self.pHypW.initialize()
        hyp_prod = self.pHypW.samples*self.pSharedHyp.samples.reshape(-1,1)
        self.pW.initialize(hyp_prod)
        self.pH.initialize(self.pSharedHyp.samples)

    def _new_params(self, n_new_features):
        new_H = np.zeros([self.D, n_new_features])
        new_Wn = np.zeros(n_new_features)
        shared_hyperprior = np.zeros(n_new_features)
        hyperprior_W = np.zeros(n_new_features)

        for k in range(n_new_features):
            hyp_prior_shared = self.pSharedHyp.prior.rvs()
            shared_hyperprior[k] = hyp_prior_shared
            hyp_prior_W = self.pHypW.prior.rvs()
            hyperprior_W[k] = hyp_prior_W
            # loop through the rows and assign samples
            new_Wn[k] = truncated_sampler(mu=0,
                                  std=1/np.sqrt(hyp_prior_shared*hyp_prior_W),
                                  lower_bound=self.trunc_low_bound,
                                  upper_bound=self.trunc_up_bound)
            for d in range(self.D):
                new_H[d,k] = truncated_sampler(mu=0,
                                              std=1/np.sqrt(hyp_prior_shared),
                                              lower_bound=self.trunc_low_bound,
                                              upper_bound=self.trunc_up_bound)


        return new_H, new_Wn, (shared_hyperprior, hyperprior_W)

    def _add_new_W_hyperpriors(self, new_Wn, new_hyperpriors, k_new, n):
        new_hyp_shared, new_hyp_Wn = new_hyperpriors
        self.pSharedHyp.add_new_features(new_hyp_shared, k_new)
        add_W = np.empty((k_new, self.N))
        add_hyp_W = add_W.copy()
        for n in range(self.N):
            for k in range(k_new):
                hyp_prior = self.pHypW.prior.rvs()
                add_hyp_W[k,n] = hyp_prior
                param = hyp_prior*new_hyp_shared[k]
                add_W[k,n] = truncated_sampler(mu=0,
                                              std=1/np.sqrt(param),
                                              lower_bound=self.trunc_low_bound,
                                              upper_bound=self.trunc_up_bound)
        add_hyp_W[:,n] = new_hyp_Wn
        add_W[:,n] = new_Wn
        self.pW.add_new_features(add_W, k_new, n)
        self.pHypW.add_new_features(add_hyp_W, k_new)

    def _sample_WH(self):
        hyp_prod = self.pHypW.samples*self.pSharedHyp.samples.reshape(-1,1)
        self.pW.update_and_sample(self.X, self.pH.samples, self.pZ.samples,
                                  hyp_prod, self.pNoise.samples)
        self.pH.update_and_sample(self.X, self.pW.samples*self.pZ.samples,
                                  self.pSharedHyp.samples, self.pNoise.samples)
    def _sample_hyperpriors(self):
        self.pSharedHyp.update_and_sample(self.pH.samples, self.pW.samples,
                                          self.pHypW.samples)
        self.pHypW.update_and_sample(self.pW.samples, self.pSharedHyp.samples)
        result = {"hyp_shared": self.pSharedHyp.samples,
                  "hyp_W": self.pHypW.samples}
        return result

    def _prune_hyperpriors(self, mask, K_plus):
        self.pSharedHyp.prune_features(mask, K_plus)
        self.pHypW.prune_features(mask, K_plus)

    def _log_joint(self):
        log_likelihood = self._log_likelihood(self.pH.samples,
                                              self.pW.samples*self.pZ.samples,
                                              self.pNoise.samples)
        #hyp_prod = self.pHypW.samples*self.pSharedHyp.samples.reshape(-1,1)
        #log_prior = self._log_prior(self.pSharedHyp.samples, hyp_prod)
        #log_hyper_prior = self._log_hyperprior()
        #log_noise = self._log_noise(self.pNoise.samples)
        #log_IBP = self._log_IBP()

        return log_likelihood #+ log_prior + log_hyper_prior + log_IBP

    def _log_prior(self, hyperprior_H, hyperprior_W):
        logpdf = trunc_norm_sum_log_pdf(self.pW.samples, hyperprior_W)
        logpdf += trunc_norm_sum_log_pdf(self.pH.samples, hyperprior_H)
        return np.sum(logpdf)

    def _log_hyperprior(self):
        logpdf = gamma_sum_log_pdf(self.pSharedHyp.samples, self.c, self.d)
        logpdf += gamma_sum_log_pdf(self.pHypW.samples, self.e, self.f)
        return np.sum(logpdf)

    def _log_IBP(self):
        log_pdf = IBP_sum_log_pdf(self.pZ.samples, self.alpha, self.beta,
                                  self.pZ.K_plus, self.pZ.M, self.N)
        return log_pdf
