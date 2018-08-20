import numpy as np
import sys
import time
from scipy.stats import gamma, bernoulli, poisson
from npbNMF.datatools import Nth_harmonic, beta_factor_logproduct, logr_z_kn,\
                             truncated_sampler

##############################################################################

# factor objects for the model - ONLY for active features

# H, W, Z

##############################################################################

class FactorZ:

    def __init__(self, D, N, K_init):

        self.N = N
        self.D = D
        self.K = K_init
        self.samples = np.zeros((self.K, self.N))
        self.M = np.sum(self.samples, axis=1)
        self.K_plus = np.count_nonzero(self.M)
        self.K = self.samples.shape[0]
        mask = self.M > 0
        self.active_mask = mask

    def update_and_sample_zn(self, Xn, H, Wn, alpha, beta, noise, n):

        mask = np.asarray([True]*self.K)
        zn = self.samples[:,n]
        m = self.M - zn

        for k in np.random.permutation(self.K):
            # if m = 0 => p(zkn=1|Z)=0, thus we can skip any calculations
            if m[k] != 0:
                mask[k] = False
                z_tilde = zn*Wn
                w_kn = Wn[k]
                logr = logr_z_kn(m[k], z_tilde, w_kn, H, Xn, noise, beta,
                                 self.N, mask)

                # logaddexp(a,b)=log(exp(a) + exp(b)) - can safely
                # calculate when a is large and b is relatively small
                # ESSENTIAL FOR MAKING THIS WORK!
                logprob = logr - np.logaddexp(logr, 0)

                samples = self.sampling(np.exp(logprob))
                self.samples[k,n] = samples
            else:
                self.samples[k,n] = 0

            mask[k] = True
        self.M = np.sum(self.samples, axis=1)
        self.K_plus = np.count_nonzero(self.M)

    def sampling(self, p):
        return bernoulli.rvs(p)

    def add_Zn_samples(self, k_new, n):
        add_Z = np.zeros([k_new, self.N])
        add_Z[:,n] = 1
        self.samples = np.append(self.samples, add_Z, axis=0)
        self.K += k_new
        # this means new features was samples, hence we use zero padding
        self.M = np.sum(self.samples, axis=1)
        self.K_plus = np.count_nonzero(self.M)


    def initialize(self):
        self.samples = np.zeros((self.K, self.N))\
                       + bernoulli.rvs(p=0.5,
                                       size=self.N*self.K).reshape(self.K,
                                                                   self.N)
        self.M = np.sum(self.samples, axis=1)
        self.K_plus = np.count_nonzero(self.M)

    def prune_features(self):
        self.active_mask = self.M > 0
        self.samples = self.samples[self.active_mask,:]
        self.M = self.M[self.active_mask]
        self.K_plus = np.count_nonzero(self.M)
        self.K = self.K_plus

class TruncNorm:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sampling(self, mu, var):
        samples = truncated_sampler(mu, np.sqrt(var),
                                    self.lower_bound, self.upper_bound)
        return samples

class FactorH(TruncNorm):

    def __init__(self, lower_bound, upper_bound, D, K_init):
        super().__init__(lower_bound, upper_bound)
        self.K = K_init
        self.D = D
        self.samples = np.empty([self.D, self.K])

    def update_and_sample(self, X, WZ, hyperprior, noise):

        WZWZ_zero_diag = WZ @ WZ.T
        np.fill_diagonal(WZWZ_zero_diag, 0)
        WZX = WZ @ X.T

        for k in np.random.permutation(self.K):
            sigma_sq = np.reciprocal(noise*np.sum(WZ[k,:]**2)
                                     + hyperprior[k])
            prod_mix =  WZX[k,:] - WZWZ_zero_diag[k,:] @ self.samples.T
            mu =  sigma_sq*(noise*prod_mix)

            # sample from truncated
            self.samples[:,k] = self.sampling(mu, sigma_sq)

    def add_new_features(self, new_features, k_new):
        self.samples = np.append(self.samples, new_features, axis=1)
        self.K = k_new + self.K

    def initialize(self, hyperprior):
        for k, param in enumerate(hyperprior):
            for d in range(self.D):
                self.samples[d,k] = truncated_sampler(mu=0,
                                                std=1/np.sqrt(param),
                                                lower_bound=self.lower_bound,
                                                upper_bound=self.upper_bound)

    def prune_features(self, mask, K_plus):
        self.samples = self.samples[:,mask]
        self.K = K_plus

class FactorW(TruncNorm):

    def __init__(self, lower_bound, upper_bound, N, K_init):
        super().__init__(lower_bound, upper_bound)
        self.N = N
        self.K = K_init
        self.samples = np.empty([self.K, self.N])

    def update_and_sample(self, X, H, Z, hyperpriors, noise):
        # now there is one sigma_sq for EACH W_nk

        if hyperpriors.ndim <= 1:
            hyperpriors = hyperpriors.reshape(-1, 1)

        HH_zero_diag = H.T @ H
        np.fill_diagonal(HH_zero_diag, 0)
        ZHX = Z*(H.T @ X)

        for k in np.random.permutation(self.K):
            prod_mix = ZHX[k,:] - Z[k,:]*(HH_zero_diag[k,:]
                                            @ (Z*self.samples))
            sigma_sq = np.reciprocal(noise*(Z[k,:])*np.sum(H[:,k]**2)
                                     + hyperpriors[k,:])
            mu =  sigma_sq*(noise*prod_mix)
            # sample from truncated
            self.samples[k,:] = self.sampling(mu, sigma_sq)

    def add_new_features(self, new_W, k_new, n):
        self.samples = np.append(self.samples, new_W, axis=0)
        self.K = k_new + self.K

    def initialize(self, hyperprior):
        if hyperprior.ndim == 1:
            hyperprior = np.broadcast_to(hyperprior.reshape(self.K, 1),
                                      (self.K, self.N))
        for k in range(self.K):
            self.samples[k,:] = truncated_sampler(mu=0,
                                                std=1/np.sqrt(hyperprior[k,:]),
                                                lower_bound=self.lower_bound,
                                                upper_bound=self.upper_bound)

    def prune_features(self, mask, K_plus):
        self.samples = self.samples[mask,:]
        self.K = K_plus

##############################################################################

# prior objects for the model

# noise, factor, and IBP priors

##############################################################################

class NoisePrior:

    def __init__(self, data_size, X, noise_alpha, noise_beta):
        self.noise_alpha = noise_alpha
        self.noise_beta = noise_beta
        self.prior = gamma(a=noise_alpha, scale=1/noise_beta)
        self.X = X
        self.trace_XX = X.ravel() @ X.ravel()

        self.alpha = noise_alpha + 0.5*data_size

    def update_and_sample(self, L, R):
        # R = W*Z
        tmp = self.trace_XX + (L.T @ L).ravel() @ (R @ R.T).ravel() \
              - 2*self.X.ravel() @ (L @ R).ravel()
        # update beta (alpha needs only one update - see __init__())
        beta = self.noise_beta + 0.5*tmp
        self.samples = self.sampling(beta)

    def sampling(self, beta):
        return gamma.rvs(a=self.alpha, scale=1/beta)

    def initialize(self):
        self.samples = self.prior.rvs()

class BaseHyperpriorShared:

    def __init__(self, alpha, beta, K=1):

        self.K = K
        self.hyperprior_alpha = alpha
        self.hyperprior_beta = beta
        self.prior = gamma(a=alpha, scale=1/beta)
        self.samples = np.empty(self.K)

    def initialize(self):
        for k in range(self.K):
            self.samples[k] = self.prior.rvs()

    def sampling(self, alpha, beta):
        for k in range(self.K):
            self.samples[k] = gamma.rvs(a=alpha, scale=1/beta[k])

    def add_new_features(self, new_features, k_new):
        self.samples = np.append(self.samples, new_features)
        self.K += k_new

    def prune_features(self, mask, K_plus):
        self.samples = self.samples[mask]
        self.K = K_plus

class HyperpriorShared(BaseHyperpriorShared):

    def __init__(self, alpha, beta, D, N, K=1):
        super().__init__(alpha, beta, K)
        self.alpha = alpha + 0.5*(D + N)

    def update_and_sample(self, H, W):
        beta = self.hyperprior_beta + 0.5*(np.sum(H**2, axis=0) + np.sum(W**2,
                                                                       axis=1))
        self.sampling(self.alpha, beta)

class HyperpriorH(BaseHyperpriorShared):

    def __init__(self, alpha, beta, D, K=1):
        super().__init__(alpha, beta, K)
        self.alpha = alpha + 0.5*D

    def update_and_sample(self, H):
        # update beta (alpha needs only one update - see __init__())
        beta = self.hyperprior_beta + 0.5*np.sum(H**2, axis=0)
        self.sampling(self.alpha, beta)

class HyperpriorSharedWithSparsity(BaseHyperpriorShared):

    def __init__(self, alpha, beta, D, N, K=1):
        super().__init__(alpha, beta, K)
        self.N = N
        self.D = D
        self.alpha = alpha + 0.5*(D + N)

    def update_and_sample(self, H, W, sparse_hyperprior):
        beta = self.hyperprior_beta + 0.5*(np.sum(H**2, axis=0)
                                           + np.sum((W**2)*sparse_hyperprior,
                                                                       axis=1))
        self.sampling(self.alpha, beta)

class BaseHyperpriorSparse:

    def __init__(self, alpha, beta, N, K=1):
        self.N = N
        self.K = K
        self.hyperprior_alpha = alpha
        self.hyperprior_beta = beta
        self.prior = gamma(a=alpha, scale=1/beta)
        self.samples = np.empty([K, N])

    def initialize(self):
        for n in range(self.N):
            for k in range(self.K):
                self.samples[k,n] = self.prior.rvs()

    def sampling(self, alpha, beta):
        for n in range(self.N):
            for k in range(self.K):
                self.samples[k,n] = gamma.rvs(a=alpha, scale=1/beta[k,n])

    def add_new_features(self, new_features, k_new):
        self.samples = np.append(self.samples, new_features, axis=0)
        self.K += k_new

    def prune_features(self, mask, K_plus):
        self.samples = self.samples[mask,:]
        self.K = K_plus

class HyperpriorSparse(BaseHyperpriorSparse):

    def __init__(self, alpha, beta, N, K=1):
        super().__init__(alpha, beta, N, K)
        self.alpha = alpha + 0.5

    def update_and_sample(self, W):
        beta = self.hyperprior_beta + 0.5*(W**2)
        self.sampling(self.alpha, beta)

class HyperpriorSparseWithShared(BaseHyperpriorSparse):

    def __init__(self, alpha, beta, N, K=1):
        super().__init__(alpha, beta, N, K)
        self.alpha = alpha + 0.5

    def update_and_sample(self, W, shared_hyperprior):
        beta = self.hyperprior_beta + 0.5*(W**2
                                         *shared_hyperprior.reshape(self.K, 1))
        self.sampling(self.alpha, beta)

class IBPAlphaPrior:

    def __init__(self, e, f, N):
        self.e = e
        self.f = f
        self.N = N
        self.prior = gamma(a=self.e, scale=1/self.f)

    def update_and_sample(self, beta, K_plus):
        """ Can be sampled using simple Gibbs sampler """

        beta_sum = Nth_harmonic(beta, self.N)
        self.samples = gamma.rvs(K_plus + self.e, scale=1/(beta_sum+self.f))

    def initialize(self):
        self.samples = self.prior.rvs()

class IBPBetaPrior:

    def __init__(self, g, h, N):
        self.g = g
        self.h = h
        self.N = N
        self.prior = gamma(a=self.g, scale=1/self.h)
        self.MHsteps = 100

    def update_and_sample(self, alpha, K_plus, M):
        """ For the IBP beta parameters a MH sampler is used

            as the gibbs sampler is infeasible to sample from

            - calculated in the log space

        """
        mask = M > 0
        beta_old = self.prior.rvs()
        beta_logprod_old = beta_factor_logproduct(beta_old, K_plus, self.N,
                                                  M[mask])
        harmonic_old = Nth_harmonic(beta_old, self.N)

        for _ in range(self.MHsteps):
            # suggest a sample
            beta_sample = self.prior.rvs()
            beta_logprod_new = beta_factor_logproduct(beta_sample, K_plus,
                                                      self.N, M[mask])
            harmonic_new = Nth_harmonic(beta_sample, self.N)

            logr = K_plus*(np.log(beta_sample) - np.log(self.samples))
            logr = logr + beta_logprod_new - beta_logprod_old
            logr += -self.g*(harmonic_new - harmonic_old)

            if logr > 0:
                beta_old = beta_sample
                beta_logprod_old = beta_logprod_new
                harmonic_old = harmonic_new
            else:
                if np.random.uniform(size=1) < np.exp(logr):
                    beta_old = beta_sample
                else:
                    pass
        self.samples = beta_old

    def initialize(self):
        self.samples = self.prior.rvs()
