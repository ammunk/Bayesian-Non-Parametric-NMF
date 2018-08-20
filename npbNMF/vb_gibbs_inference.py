##############################################################################

# MCMC & VB COMBINED INFERENCE
# Infer p(Z|X) using Gibbs sampling, but approximate parameter marginalization
# using VB

##############################################################################

import time
import sys
import copy
import numpy as np
from scipy.stats import bernoulli, poisson
from npbNMF.datatools import mean_X_LR_error_fast, IBP_sum_log_pdf
class track_Gibbs_VB_Z:

    def __init__(self, K_init, N):
        """
        pZs is a list of FactorZ objects.

        This class keeps track of number of active features, and
        concatenates all samples from pZs to build the matrix Z.

        This is done numerous times throughout the Gibbs sampler

        """
        self.N = N
        self.K_plus = K_init
        self.samples = self._initialize()

    def add_new_samples(self, new_Z):
        self.samples = new_Z
        self.M = np.sum(self.samples, axis=1)
        self.K_plus = new_Z.shape[0]

    def _initialize(self):
        # initialize all z_kn, with probability p=0.5 of being active
        samples = bernoulli.rvs(p=0.5,
                                size=self.N*self.K_plus).reshape(self.K_plus,
                                                                        self.N)
        self.M = np.sum(samples, axis=1)
        self.K_plus = np.count_nonzero(self.M)
        mask = self.M > 0
        self.active_mask = mask
        return samples

    def prune_features(self):
        self.M = np.sum(self.samples, axis=1)
        self.active_mask = self.M > 0
        self.samples = self.samples[self.active_mask,:]
        self.M = self.M[self.active_mask]
        self.K_plus = np.count_nonzero(self.M)
        return self.active_mask

class BaseGibbsVB:

    def __init__(self, X, K_init, tolerance, max_iter, a, b, alpha, beta,
                 num_samples=2000):
        from npbNMF.gibbs_vb_distributions import HFactor, WFactor,\
                                                  NoiseGamma
        self.X = X
        self.K_init = K_init
        self.D, self.N = self.X.shape
        self.data_size = self.D*self.N
        self.trained = False

        self.num_samples = num_samples # sample size
        self.max_iter = max_iter # number of iterations for VB step
        self.tolerance = tolerance

        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta

        # build basic model components and initialize them
        self.pZ = track_Gibbs_VB_Z(K_init, self.N)
        self.q_W = WFactor(0, np.infty, self.N, K_init)
        self.q_H = HFactor(0, np.infty, self.D, K_init)
        self.q_noise = NoiseGamma(self.X, self.a, self.b, self.data_size)
        self.trained = False

    def train(self, info_dict, log):

        log = log
        min_iter = self.max_iter/ self.N
        n_iterations = min_iter

        i = 0
        log.info("Initializing")
        self.ELBO, ELBO_lik = self._initialize(self.K_init)
        log.info("Finished initializing - beginning training")
        self.samples = {"Z": [],
                        "active_features": [], # integer, thus no copy
                        "M": []}

        info_dict["log_joint"] = []
        info_dict["K_feature_change"] = []
        info_dict["K_plus"] = []
        info_dict["weighted_K_plus"] = []
        info_dict["n_features"] = []
        info_dict["weighted_n_features"] = []
        K_plus_old = self.K_init
        n_predict = 0
        burnin = 100
        expected_predicted = 0

        while i < self.num_samples:
            timer = time.time()
            samples = self._sample(self.ELBO, ELBO_lik)
            sampling_time = time.time() - timer
            timer = time.time()
            self.ELBO, ELBO_lik = self._full_VB_train()
            VB_train_time = time.time() - timer
            logpdf_Z = IBP_sum_log_pdf(self.pZ.samples, self.alpha, self.beta,
                                       self.pZ.K_plus, self.pZ.M, self.N)
            log_joint = self.ELBO + logpdf_Z

            info_dict["log_joint"].append(log_joint)
            if self.pZ.K_plus != K_plus_old:
                info_dict["K_feature_change"].append(i)
                K_plus_old = self.pZ.K_plus
            info_dict["K_plus"].append(self.pZ.K_plus)
            info_dict["weighted_K_plus"].append(int(np.sum(np.sum(self.pZ.samples
                                                       *self.q_W.mean,
                                                       axis=1)>1e-16)))
            info_dict["n_features"].append(np.sum(self.pZ.samples,
                                                  axis=0).tolist())
            info_dict["weighted_n_features"].append(np.sum(self.pZ.samples
                                                           *self.q_W.mean
                                                           >1e-16,
                                                           axis=0).tolist())
            self.store_samples(samples)
            sum_sq_error = np.sum((self.X-self.predict())**2)
            msg = (f"\n\tIteration = {i}")
            msg += (f"\n\t- Active features = {self.pZ.K_plus}")
            msg += (f"\n\t- M = {self.pZ.M}")
            msg += (f"\n\t- Sampling time = {sampling_time}")
            msg += "\n-----------------------------------------------\n"
            msg += (f"After full VB training")
            msg += (f"\n\t- Time: {VB_train_time}")
            msg += (f"\n\t- Sq Error = {sum_sq_error}")
            msg += (f"\n\t- Final ELBO = {self.ELBO}")
            msg += "\n===============================================\n"
            log.info(msg)
            if i >= burnin:
                expected_predicted += self.predict()
                n_predict += 1
            i += 1
        expected_predicted = expected_predicted/n_predict
        info_dict["sq_error"] = np.sum((self.X-expected_predicted)**2)
        self.trained = True

    def _initialize(self, K_init):
        self._initialize_WH(K_init)
        # update hyperparameters
        self._update_hyperpriors()
        # noise update
        self.q_noise.update(self.q_H.mean, self.q_H.mean_sq,
                            self.q_W.mean*self.pZ.samples,
                            self.q_W.mean_sq*self.pZ.samples)
        return self._ELBO()

    def _initialize_WH(self, K_init):
        pass

    def _full_VB_train(self):

        ELBO_old = -1e21
        rel_ELBO = np.inf
        i = 0
        while rel_ELBO > self.tolerance and i < self.max_iter:
            feature_update_list = np.random.permutation(self.pZ.K_plus)
            self._full_update(feature_update_list)
            # calculate ELBO, which we seek to maximize
            # maximizing the ELBO corresponds to minimizing the
            # KL divergence
            ELBO, ELBO_lik = self._ELBO()
            ELBO_diff = ELBO - ELBO_old
            rel_ELBO = ELBO_diff / abs(ELBO_old)
            ELBO_old = ELBO
            i = i + 1
            break
        return ELBO, ELBO_lik

    def _sample(self, ELBO, ELBO_lik):

        for n in np.random.permutation(self.N):
            m = self.pZ.M - self.pZ.samples[:,n]
            old_elbo, old_Z, old_lik_elbo = self._sample_Zn(
                                                        self.pZ.samples.copy(),
                                                        m, n, ELBO, ELBO_lik)
            new_Z, ELBO, ELBO_lik = self._sample_Kn(old_elbo, old_Z, n,
                                                    old_lik_elbo)
            self.pZ.add_new_samples(new_Z)
        self.prune_features()

        return {"Z": self.pZ.samples.copy(),
                "active_features": self.pZ.K_plus, # integer, thus no copy
                "M": self.pZ.M.copy()}

    def store_samples(self, samples):
        for key, sample in samples.items():
            self.samples[key].append(sample)

    def _sample_Zn(self, Z_old, m, n, old_elbo, old_lik_elbo):
        tmp_Z = Z_old.copy()
        noise_mean, ln_noise_mean = self.q_noise.mean,\
                                    self.q_noise.ln_mean
        const_elbo = self._const_elbo_zn()
        for k in np.random.permutation(self.pZ.K_plus):
            feature_update_list = np.arange(1) + k
            if Z_old[k,n] == 1:
                const_lik_elbo = old_lik_elbo\
                                 - self._ELBO_likelihood_Wpart(Z_old, k, n,
                                                               self.pZ.K_plus)
            else:
                const_lik_elbo = old_lik_elbo
            const_elbo += const_lik_elbo
            if m[k] != 0:
                old_W = self.q_W.get_attributes()
                old_hyperprior_W = self._get_hyp_W_attributes()
                old_Z_kn = Z_old[k,n]
                if old_Z_kn == 0:
                    tmp_Z[k,n] = 1
                else:
                    tmp_Z[k,n] = 0
                ELBO_part_old = -1e21
                rel_ELBO = np.inf
                i = 0
                while rel_ELBO > self.tolerance and i < self.max_iter:
                    self._update_W_parts(tmp_Z, feature_update_list, n)
                    ELBO_part_prior = self._elbo_parts_W()
                    if tmp_Z[k,n] == 1:
                        ELBO_lik_part = self._ELBO_likelihood_Wpart(tmp_Z, k,
                                                             n, self.pZ.K_plus)
                    else:
                        ELBO_lik_part = 0
                    ELBO_part = ELBO_part_prior + ELBO_lik_part
                    ELBO_diff = ELBO_part - ELBO_part_old
                    rel_ELBO = ELBO_diff / abs(ELBO_part_old + const_elbo)
                    ELBO_part_old = ELBO_part
                    i += 1
                new_elbo = ELBO_part_old + const_elbo
                new_lik_elbo = const_lik_elbo + ELBO_lik_part
                log_r_p = np.log(m[k]) - np.log(self.beta + self.N - 1 - m[k])
                if old_Z_kn == 1:
                    logr = old_elbo - new_elbo
                    logr += log_r_p
                    log_p = logr - np.logaddexp(logr, 0)
                else:
                    logr = new_elbo - old_elbo
                    logr += log_r_p
                    log_p = logr - np.logaddexp(logr, 0)
                z_kn = bernoulli.rvs(p=np.exp(log_p))
                if z_kn == old_Z_kn:
                    self.q_W.set_attributes(old_W)
                    self._set_hyperprior_W_attributes(old_hyperprior_W)
                else:
                    Z_old[k,n] = z_kn
                    old_elbo = new_elbo
                    old_lik_elbo = new_lik_elbo
            else:
                if Z_old[k,n] != 0:
                    Z_old[k,n] = 0
                    ELBO_part_old = -1e21
                    rel_ELBO = np.inf
                    i = 0
                    while rel_ELBO > self.tolerance and i < self.max_iter:
                        self._update_W_parts(Z_old, feature_update_list, n)
                        ELBO_part_prior = self._elbo_parts_W()
                        if tmp_Z[k,n] == 1:
                            ELBO_lik_part = self._ELBO_likelihood_Wpart(tmp_Z,
                                                          k, n, self.pZ.K_plus)
                        else:
                            ELBO_lik_part = 0
                        ELBO_part = ELBO_part_prior + ELBO_lik_part
                        ELBO_diff = ELBO_part - ELBO_part_old
                        rel_ELBO = ELBO_diff / abs(ELBO_part_old + const_elbo)
                        ELBO_part_old = ELBO_part
                        i += 1
                    old_elbo = ELBO_part + const_elbo
                    old_lik_elbo = const_lik_elbo + ELBO_lik_part
                else:
                    old_elbo = old_elbo
                    old_lik_elbo = old_lik_elbo
        return old_elbo, Z_old, old_lik_elbo

    def _get_hyp_W_attributes(self):
        pass

    def _set_hyperprior_W_attributes(self, old_hyperprior_W):
        pass

    def _const_elbo_zn(self):
        pass

    def _update_W_parts(self, Z, feature_update_list, n):
        pass

    def _elbo_parts_W(self):
        pass

    def _sample_Kn(self, old_elbo, old_Z, n, old_lik_elbo):
        K_n = poisson.rvs(self.alpha*self.beta/(self.beta+self.N-1))
        old_W = self.q_W.get_attributes()
        old_H = self.q_H.get_attributes()
        old_hyperpriors = self._get_hyperpriors_attributes()

        if K_n > 0 :
            self._add_new_features(K_n)
            tmp_Z = np.append(old_Z, np.zeros([K_n,self.N]), axis=0)
            tmp_Z[-K_n:,n] = 1
            feature_update_list = np.arange(K_n) + self.pZ.K_plus
            ELBO_part_old = -1e21
            const_elbo = self.q_noise.elbo_part()
            noise_mean, ln_noise_mean = self.q_noise.mean, self.q_noise.ln_mean
            rel_ELBO = np.inf
            i = 0
            while rel_ELBO > self.tolerance and i < self.max_iter:
                self._update_all_but_noise(tmp_Z, feature_update_list, n)
                ELBO_part_prior = self._elbo_parts_all_but_noise()
                ELBO_lik = self._ELBO_likelihood(self.q_H.mean,
                                                 self.q_H.mean_sq,
                                                 self.q_W.mean*tmp_Z,
                                                 self.q_W.mean_sq*tmp_Z,
                                                 noise_mean,
                                                 ln_noise_mean)
                ELBO_part = ELBO_part_prior + ELBO_lik
                ELBO_diff = ELBO_part - ELBO_part_old
                rel_ELBO = ELBO_diff / abs(ELBO_part_old + const_elbo)
                ELBO_part_old = ELBO_part
                i += 1
            new_elbo = ELBO_part_old + const_elbo

            logr = new_elbo - old_elbo
            if (logr > 0) or (np.random.uniform() < np.exp(logr)):
                new_lik_elbo = ELBO_lik
            else:
                tmp_Z = old_Z
                self.q_W.set_attributes(old_W)
                self.q_H.set_attributes(old_H)
                self._set_hyperpriors_attributes(old_hyperpriors)
                new_lik_elbo = old_lik_elbo
        else:
            tmp_Z = old_Z
            new_elbo = old_elbo
            new_lik_elbo = old_lik_elbo
        return tmp_Z, new_elbo, new_lik_elbo

    def _get_hyperpriors_attributes(self):
        pass

    def _set_hyperpriors_attributes(self, old_hyperpriors):
        pass

    def _add_new_features(self, K_n):
        pass

    def _update_all_but_noise(self, Z, feature_update_list, n):
        pass

    def _elbo_parts_all_but_noise(self):
        pass

    def _full_update(self, feature_update_list):
        # update factors (W and H) one after the other
        self._update_WH(feature_update_list)
        self._update_hyperpriors()
        # noise update
        self.q_noise.update(self.q_H.mean, self.q_H.mean_sq,
                            self.q_W.mean*self.pZ.samples,
                            self.q_W.mean_sq*self.pZ.samples)

    def _update_WH(self, feature_update_list):
        pass

    def _update_hyperpriors(self):
        pass

    def prune_features(self):
        mask = self.pZ.prune_features()
        self.q_W.prune_features(mask)
        self.q_H.prune_features(mask)
        self._prune_hyperpriors(mask)

    def _prune_hyperpriors(self, mask):
        pass

    def _ELBO(self):
        noise_mean, ln_noise_mean = self.q_noise.mean, self.q_noise.ln_mean
        elbo_likelihood = self._ELBO_likelihood(self.q_H.mean,
                                              self.q_H.mean_sq,
                                              self.q_W.mean*self.pZ.samples,
                                              self.q_W.mean_sq*self.pZ.samples,
                                              noise_mean,
                                              ln_noise_mean)
        elbo_priors = self._ELBO_priors()
        return elbo_likelihood + elbo_priors, elbo_likelihood

    def _ELBO_priors(self):
        elbo_noise = self.q_noise.elbo_part()
        part_elbo = self._elbo_parts_all_but_noise()
        return elbo_noise + part_elbo

    def _elbo_priors_hyperpriors(self):
        pass

    def _ELBO_likelihood(self, mean_H, mean_sq_H, mean_WZ, mean_sq_WZ,
                         noise_mean, ln_noise_mean):
        mean_sq_sum_H = np.sum(mean_sq_H, axis=0)
        mean_sq_sum_WZ = np.sum(mean_sq_WZ, axis=1)
        sum_mean_sq_error = mean_X_LR_error_fast(self.X, mean_H,
                                                  mean_sq_sum_H, mean_WZ,
                                                  mean_sq_sum_WZ)

        ELBO_lik = self.data_size*0.5*(ln_noise_mean - np.log(2*np.pi))
        ELBO_lik = ELBO_lik - 0.5*noise_mean*(sum_mean_sq_error)
        return ELBO_lik

    def _ELBO_likelihood_Wpart(self, Z, k, n, K):
        part = -2*self.q_W.mean[k,n]*(self.X[:,n].reshape(1,self.D)
                                      @ self.q_H.mean[:,k].reshape(self.D,1))
        part += self.q_W.mean_sq[k,n]*(np.sum(self.q_H.mean_sq[:,k]))
        mask = np.asarray([True]*K)
        mask[k] = False
        part += self.q_W.mean[k,n]*self.q_H.mean[:,k].reshape(1,self.D)\
                            @(self.q_H.mean[:,mask]
                            @(self.q_W.mean[mask,n]
                              *Z[mask,n]).reshape(K-1,1)).reshape(self.D,1)
        part = -0.5*self.q_noise.mean*part
        return part

    def predict(self):
        return self.q_H.mean @ (self.q_W.mean*self.pZ.samples)

    def extract_features(self):
        return self.q_H.mean

    def transform(self, X=None):
        import sys
        if self.trained:
            if X:
                pass
            else:
                return self.q_W.mean*self.pZ.samples
        else:
            sys.exit("ERROR - MODEL NOT TRAINED")

    def pickle_factors(self, filename):
        import pickle
        feature_dict = {'H': self.q_H.mean,
                        'W': self.q_W.mean,
                        'Z': self.pZ.samples}
        with open(filename, 'bw') as f:
            pickle.dump(feature_dict, f)

class GibbsVBSharedNPB(BaseGibbsVB):

    def __init__(self, X, K_init, tolerance, max_iter, a, b, c, d, alpha, beta,
                 num_samples=2000):
        super().__init__(X, K_init, tolerance, max_iter, a, b, alpha, beta,
                         num_samples=num_samples)
        from npbNMF.gibbs_vb_distributions import HyperpriorShared

        self.c = c
        self.d = d
        self.q_hyperprior = HyperpriorShared(c, d, self.D, self.N, K_init)

    def _update_WH(self, feature_update_list):
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq,
                        self.pZ.samples, self.q_hyperprior.mean,
                        self.q_noise.mean, feature_update_list)
        self.q_H.update(self.X, self.q_W.mean*self.pZ.samples,
                        self.q_W.mean_sq*self.pZ.samples,
                        self.q_hyperprior.mean, self.q_noise.mean,
                        feature_update_list)

    def _initialize_WH(self, K_init=1):
        expected_param = self.c/self.d
        expected_param_inv = 1/expected_param
        self.q_W.initialize(expected_param_inv, K_init)
        self.q_H.initialize(expected_param_inv, K_init)

    def _update_hyperpriors(self):
        self.q_hyperprior.update(self.q_H.mean_sq, self.q_W.mean_sq)

    def _get_hyp_W_attributes(self):
        return None

    def _set_hyperprior_W_attributes(self, old_hyperprior_W=None):
        pass

    def _get_hyperpriors_attributes(self):
        return self.q_hyperprior.get_attributes()

    def _set_hyperpriors_attributes(self, old_hyperpriors):
        self.q_hyperprior.set_attributes(old_hyperpriors)

    def _add_new_features(self, K_n):
        expected_param = self.c/self.d
        expected_param_inv = 1/expected_param
        self.q_H.add_new_features(expected_param_inv, K_n)
        self.q_W.add_new_features(expected_param_inv, K_n)
        self.q_hyperprior.add_new_features(K_n)

    def _update_all_but_noise(self, Z, feature_update_list, n):
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq,
                        Z, self.q_hyperprior.mean,
                        self.q_noise.mean, feature_update_list, n)
        self.q_H.update(self.X, self.q_W.mean*Z, self.q_W.mean_sq*Z,
                        self.q_hyperprior.mean, self.q_noise.mean,
                        feature_update_list)
        self.q_hyperprior.update(self.q_H.mean_sq, self.q_W.mean_sq)

    def _elbo_parts_all_but_noise(self):
        elbo_W = self.q_W.elbo_part(self.q_hyperprior.mean,
                                    self.q_hyperprior.ln_mean)
        elbo_H = self.q_H.elbo_part(self.q_hyperprior.mean,
                                    self.q_hyperprior.ln_mean)
        elbo_hyperprior = self.q_hyperprior.elbo_part()
        return elbo_W + elbo_H + elbo_hyperprior

    def _const_elbo_zn(self):
        const_elbo = self.q_noise.elbo_part()
        const_elbo += self.q_H.elbo_part(self.q_hyperprior.mean,
                                         self.q_hyperprior.ln_mean)
        const_elbo += self.q_hyperprior.elbo_part()
        return  const_elbo

    def _update_W_parts(self, Z, feature_update_list, n):
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq,
                        Z, self.q_hyperprior.mean, self.q_noise.mean,
                        feature_update_list, n)

    def _elbo_parts_W(self):
        elbo_W = self.q_W.elbo_part(self.q_hyperprior.mean,
                                    self.q_hyperprior.ln_mean)
        return elbo_W

    def _elbo_priors_hyperpriors(self):
        return self.q_hyperprior.elbo_part()

    def _prune_hyperpriors(self, mask):
        self.q_hyperprior.prune_features(mask)

class GibbsVBSparseNPB(BaseGibbsVB):

    def __init__(self, X, K_init, tolerance, max_iter, a, b, c, d, e, f,
                 alpha, beta, num_samples=2000):
        super().__init__(X, K_init, tolerance, max_iter, a, b, alpha, beta,
                         num_samples=num_samples)
        from npbNMF.gibbs_vb_distributions import HyperpriorH, HyperpriorSparse

        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.q_hyperprior_H = HyperpriorH(c, d, self.D, K_init)
        self.q_hyperprior_W = HyperpriorSparse(e, f, self.N, K_init)

    def _update_WH(self, feature_update_list):
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq,
                        self.pZ.samples, self.q_hyperprior_W.mean,
                        self.q_noise.mean, feature_update_list)
        self.q_H.update(self.X, self.q_W.mean*self.pZ.samples,
                        self.q_W.mean_sq*self.pZ.samples,
                        self.q_hyperprior_H.mean, self.q_noise.mean,
                        feature_update_list)

    def _initialize_WH(self, K_init=1):
        expected_param_inv_H = self.d/self.c
        expected_param_inv_W = self.f/self.e
        self.q_W.initialize(expected_param_inv_W, K_init)
        self.q_H.initialize(expected_param_inv_H, K_init)

    def _update_hyperpriors(self):
        self.q_hyperprior_W.update(self.q_W.mean_sq)
        self.q_hyperprior_H.update(self.q_H.mean_sq)

    def _update_all_but_noise(self, Z, feature_update_list, n):
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq,
                        Z, self.q_hyperprior_W.mean, self.q_noise.mean,
                        feature_update_list, n)
        self.q_H.update(self.X, self.q_W.mean*Z, self.q_W.mean_sq*Z,
                        self.q_hyperprior_H.mean, self.q_noise.mean,
                        feature_update_list)
        self.q_hyperprior_H.update(self.q_H.mean_sq)
        self.q_hyperprior_W.update(self.q_W.mean_sq)

    def _get_hyp_W_attributes(self):
        return self.q_hyperprior_W.get_attributes()

    def _set_hyperprior_W_attributes(self, old_hyperprior_W=None):
        self.q_hyperprior_W.set_attributes(old_hyperprior_W)

    def _get_hyperpriors_attributes(self):
        return [self.q_hyperprior_H.get_attributes(),
                self.q_hyperprior_W.get_attributes()]

    def _set_hyperpriors_attributes(self, old_hyperpriors):
        old_hyp_H, old_hyp_W = old_hyperpriors
        self.q_hyperprior_H.set_attributes(old_hyp_H)
        self.q_hyperprior_W.set_attributes(old_hyp_W)

    def _add_new_features(self, K_n):
        expected_param_inv_H = self.d/self.c
        expected_param_inv_W = self.f/self.e
        self.q_H.add_new_features(expected_param_inv_H, K_n)
        self.q_W.add_new_features(expected_param_inv_W, K_n)
        self.q_hyperprior_H.add_new_features(K_n)
        self.q_hyperprior_W.add_new_features(K_n)

    def _elbo_parts_all_but_noise(self):
        elbo_W = self.q_W.elbo_part(self.q_hyperprior_W.mean,
                                    self.q_hyperprior_W.ln_mean)
        elbo_H = self.q_H.elbo_part(self.q_hyperprior_H.mean,
                                    self.q_hyperprior_H.ln_mean)
        elbo_hyperprior_H = self.q_hyperprior_H.elbo_part()
        elbo_hyperprior_W = self.q_hyperprior_W.elbo_part()
        return elbo_W + elbo_H + elbo_hyperprior_H + elbo_hyperprior_W

    def _const_elbo_zn(self):
        const_elbo = self.q_noise.elbo_part()
        const_elbo += self.q_H.elbo_part(self.q_hyperprior_H.mean,
                                         self.q_hyperprior_H.ln_mean)
        const_elbo += self.q_hyperprior_H.elbo_part()
        return  const_elbo

    def _update_W_parts(self, Z, feature_update_list, n):
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq,
                        Z, self.q_hyperprior_W.mean,
                        self.q_noise.mean, feature_update_list, n)
        self.q_hyperprior_W.update(self.q_W.mean_sq)

    def _elbo_parts_W(self):
        elbo_W = self.q_W.elbo_part(self.q_hyperprior_W.mean,
                                    self.q_hyperprior_W.ln_mean)
        elbo_hyperprior_W = self.q_hyperprior_W.elbo_part()
        return elbo_W + elbo_hyperprior_W

    def _elbo_priors_hyperpriors(self):
        return self.q_hyperprior_H.elbo_part()\
               + self.q_hyperprior_W.elbo_part()

    def _prune_hyperpriors(self, mask):
        self.q_hyperprior_H.prune_features(mask)
        self.q_hyperprior_W.prune_features(mask)

class GibbsVBSharedSparseNPB(BaseGibbsVB):

    def __init__(self, X, K_init, tolerance, max_iter, a, b, c, d, e, f, alpha,
                 beta, num_samples=2000):
        super().__init__(X, K_init, tolerance, max_iter, a, b, alpha, beta,
                         num_samples=num_samples)
        from npbNMF.gibbs_vb_distributions import HyperpriorSharedWithSparse,\
                                                  HyperpriorSparseWithShared

        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.q_hyperprior_shared = HyperpriorSharedWithSparse(c, d, self.D,
                                                              self.N, K_init)
        self.q_hyperprior_W = HyperpriorSparseWithShared(e, f, self.N, K_init)

    def _update_WH(self, feature_update_list):
        prod = self.q_hyperprior_W.mean.T*self.q_hyperprior_shared.mean
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq,
                        self.pZ.samples, prod.T, self.q_noise.mean,
                        feature_update_list)
        self.q_H.update(self.X, self.q_W.mean*self.pZ.samples,
                        self.q_W.mean_sq*self.pZ.samples,
                        self.q_hyperprior_shared.mean, self.q_noise.mean,
                        feature_update_list)

    def _initialize_WH(self, K_init=1):
        expected_param_inv_shared = self.d/self.c
        expected_param_inv_W = (self.f/self.e)*(1/expected_param_inv_shared)
        self.q_W.initialize(expected_param_inv_W, K_init)
        self.q_H.initialize(expected_param_inv_shared, K_init)


    def _update_hyperpriors(self):
        self.q_hyperprior_shared.update(self.q_H.mean_sq, self.q_W.mean_sq,
                                        self.q_hyperprior_W.mean)
        self.q_hyperprior_W.update(self.q_W.mean_sq,
                                   self.q_hyperprior_shared.mean)

    def _get_hyp_W_attributes(self):
        return self.q_hyperprior_W.get_attributes()

    def _set_hyperprior_W_attributes(self, old_hyperprior_W=None):
        self.q_hyperprior_W.set_attributes(old_hyperprior_W)

    def _get_hyperpriors_attributes(self):
        return [self.q_hyperprior_shared.get_attributes(),
                self.q_hyperprior_W.get_attributes()]

    def _set_hyperpriors_attributes(self, old_hyperpriors):
        old_hyp_shared, old_hyp_W = old_hyperpriors
        self.q_hyperprior_shared.set_attributes(old_hyp_shared)
        self.q_hyperprior_W.set_attributes(old_hyp_W)

    def _add_new_features(self, K_n):
        expected_param_inv_shared = self.d/self.c
        expected_param_inv_W = (self.f/self.e)*(expected_param_inv_shared)
        self.q_H.add_new_features(expected_param_inv_shared, K_n)
        self.q_W.add_new_features(expected_param_inv_W, K_n)
        self.q_hyperprior_shared.add_new_features(K_n)
        self.q_hyperprior_W.add_new_features(K_n)

    def _update_all_but_noise(self, Z, feature_update_list, n):
        prod = self.q_hyperprior_W.mean.T*self.q_hyperprior_shared.mean
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq,
                        Z, prod.T, self.q_noise.mean,
                        feature_update_list, n)
        self.q_H.update(self.X, self.q_W.mean*Z, self.q_W.mean_sq*Z,
                        self.q_hyperprior_shared.mean, self.q_noise.mean,
                        feature_update_list)
        self.q_hyperprior_shared.update(self.q_H.mean_sq, self.q_W.mean_sq,
                                        self.q_hyperprior_W.mean)
        self.q_hyperprior_W.update(self.q_W.mean_sq,
                                   self.q_hyperprior_shared.mean)

    def _elbo_parts_all_but_noise(self):
        hyp_prod = self.q_hyperprior_W.mean.T*self.q_hyperprior_shared.mean
        hyp_sum = self.q_hyperprior_W.ln_mean.T\
                  + self.q_hyperprior_shared.ln_mean
        elbo_W = self.q_W.elbo_part(hyp_prod.T, hyp_sum.T)
        elbo_H = self.q_H.elbo_part(self.q_hyperprior_shared.mean,
                                    self.q_hyperprior_shared.ln_mean)
        elbo_hyperprior_H = self.q_hyperprior_shared.elbo_part()
        elbo_hyperprior_W = self.q_hyperprior_W.elbo_part()
        return elbo_W + elbo_H + elbo_hyperprior_H + elbo_hyperprior_W

    def _const_elbo_zn(self):
        const_elbo = self.q_noise.elbo_part()
        const_elbo += self.q_H.elbo_part(self.q_hyperprior_shared.mean,
                                         self.q_hyperprior_shared.ln_mean)
        const_elbo += self.q_hyperprior_shared.elbo_part()
        return  const_elbo

    def _update_W_parts(self, Z, feature_update_list, n):
        hyp_prod = self.q_hyperprior_W.mean.T*self.q_hyperprior_shared.mean
        self.q_W.update(self.X, self.q_H.mean, self.q_H.mean_sq,
                        Z, hyp_prod.T, self.q_noise.mean, feature_update_list,
                        n)
        self.q_hyperprior_W.update(self.q_W.mean_sq,
                                   self.q_hyperprior_shared.mean)

    def _elbo_parts_W(self):
        hyp_prod = self.q_hyperprior_W.mean.T*self.q_hyperprior_shared.mean
        hyp_sum = self.q_hyperprior_W.ln_mean.T\
                  + self.q_hyperprior_shared.ln_mean
        elbo_W = self.q_W.elbo_part(hyp_prod.T, hyp_sum.T)
        elbo_hyperprior_W = self.q_hyperprior_W.elbo_part()
        return elbo_W + elbo_hyperprior_W

    def _prune_hyperpriors(self, mask):
        self.q_hyperprior_shared.prune_features(mask)
        self.q_hyperprior_W.prune_features(mask)
