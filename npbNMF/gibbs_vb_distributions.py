import sys
import time
import numpy as np
from scipy.stats import truncnorm, gamma, expon
from scipy.special import expit, gammaln
from npbNMF.datatools import trunc_moments, means_factor_prod, expon_moments,\
                             gamma_moments, gamma_prior_elbo, \
                             mean_sq, mean_X_LR_error_fast, mean_X_LR_error

##############################################################################

# factor objects for the model - ONLY for active features

##############################################################################

class FactorTruncNormVB:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def initialize_preset(self, init_mean, init_var):

        self.mean = init_mean
        self.var = init_var
        self.mean_sq = mean_sq(self.mean, self.var)

    def moments(self, mu, sigma_sq):
        new_mean, new_var, new_etrpy = trunc_moments(
                                                     self.lower_bound,
                                                     self.upper_bound,
                                                     mu,
                                                     sigma_sq
                                                    )
        new_mean_sq = mean_sq(new_mean, new_var)

        return new_mean, new_var, new_etrpy, new_mean_sq

class HFactor(FactorTruncNormVB):

    def __init__(self, lower_bound, upper_bound, D, K_init):
        super().__init__(lower_bound, upper_bound)
        self.D = D
        self.K_init = K_init
        self.mean = np.empty([D, K_init])
        self.mean_sq = np.empty_like(self.mean)
        self.var = np.empty_like(self.mean)
        self.etrpy = np.zeros_like(self.mean)

    def initialize(self, expected_param_inv, K_init=1):
        init_mean =  np.empty([self.D, K_init])
        init_var = np.empty_like(init_mean)
        init_etrpy = np.zeros_like(init_mean)

        from npbNMF.datatools import sample_init_factor_expon
        for d in range(self.D):
            for k in range(K_init):
                estimates = self._sample_estimates(expected_param_inv)
                init_mean[d,k], init_var[d,k] = estimates
                # leave the entropy, as we refrain from estimating it

        self.mean = init_mean
        self.var = init_var
        self.etrpy = init_etrpy
        self.mean_sq = mean_sq(self.mean, self.var)

    def _sample_estimates(self, expected_var):
        from npbNMF.datatools import sample_init_factor_TN
        estimates  = sample_init_factor_TN(self.lower_bound, self.upper_bound,
                                           np.sqrt(expected_var),
                                           n_samples=10)
        return estimates

    def update(self, X, mean_WZ, mean_sq_WZ, mean_hyperprior, mean_noise,
               feature_update_list):

        mean_sq_sum_WZ = np.sum(mean_sq_WZ, axis=1)
        sigma_sq = np.reciprocal(mean_noise*mean_sq_sum_WZ + mean_hyperprior)

        WZWZ_zero_diag = mean_WZ @ mean_WZ.T
        np.fill_diagonal(WZWZ_zero_diag, 0)
        WZX = mean_WZ @ X.T

        for k in feature_update_list:
            prod_mix = WZX[k,:] -  WZWZ_zero_diag[k,:] @ self.mean.T
            mu =  sigma_sq[k]*(mean_noise*prod_mix)
            moments = self.moments(mu, sigma_sq[k])
            new_mean, new_var, new_etrpy, new_mean_sq = moments
            self.mean[:,k] = new_mean
            self.var[:,k] = new_var
            self.etrpy[:,k] = new_etrpy
            self.mean_sq[:,k] = new_mean_sq

    def elbo_part(self, mean_hyperprior, ln_mean_hyperprior):
        """ Calculate ELBO contribution

            PRIOR ON H: H ~ TruncatedNormal(mu=0, var=hyperprior,
                                            lower_bound=0,upper_bound=infity)

        INPUT
        =====
            - mean_param: array-like, shape (K)
            - ln_mean_param: array-like, shape (K)

        OUTPUT
        ======
            - elbo_part: float

        """
        transposed_tmp = ln_mean_hyperprior - self.mean_sq*mean_hyperprior
        prior_elbo = np.sum(-0.5*np.log(2*np.pi) + 0.5*transposed_tmp
                            + np.log(2))
        entropy_elbo = np.sum(self.etrpy)
        return prior_elbo + entropy_elbo

    def add_new_features(self, expected_param_inv, k_new):
        new_moments = self._initialize_new_features(expected_param_inv, k_new)
        new_mean, new_var, new_etrpy = new_moments
        new_mean_sq = mean_sq(new_mean, new_var)
        self.mean = np.append(self.mean, new_mean, axis=1)
        self.var = np.append(self.var, new_var, axis=1)
        self.mean_sq = np.append(self.mean_sq, new_mean_sq, axis=1)
        self.etrpy = np.append(self.etrpy, new_etrpy, axis=1)

    def _initialize_new_features(self, expected_param_inv, k_new):
        new_mu = np.zeros((self.D, k_new))# the prior
        sigma_sq_expected = np.full((self.D, k_new), expected_param_inv)
        new_mean = np.empty((self.D, k_new))
        new_var = new_mean.copy()
        new_etrpy = new_var.copy()
        for k in range(k_new):
            new_mean[:,k], new_var[:,k], new_etrpy[:,k] = trunc_moments(
                                                         self.lower_bound,
                                                         self.upper_bound,
                                                         new_mu[:,k],
                                                         sigma_sq_expected[:,k]
                                                                      )

        return new_mean, new_var, new_etrpy

    def prune_features(self, mask):
        self.mean = self.mean[:,mask]
        self.var = self.var[:,mask]
        self.etrpy = self.etrpy[:,mask]
        self.mean_sq = self.mean_sq[:,mask]

    def get_attributes(self):
        return (self.mean.copy(), self.var.copy(), self.mean_sq.copy(),
                self.etrpy.copy())

    def set_attributes(self, attributes):
        self.mean, self.var, self.mean_sq, self.etrpy = attributes

class WFactor(FactorTruncNormVB):

    def __init__(self, lower_bound, upper_bound, N, K_init):
        super().__init__(lower_bound, upper_bound)
        self.N = N
        self.K_init = K_init
        self.mean = np.empty([K_init,N])
        self.mean_sq = np.empty_like(self.mean)
        self.var = np.empty_like(self.mean)
        self.etrpy = np.zeros_like(self.mean)

    def initialize(self, expected_param_inv, K_init=1):
        init_mean =  np.empty([K_init, self.N])
        init_var = np.empty_like(init_mean)
        init_etrpy = np.zeros_like(init_mean)

        for n in range(self.N):
            for k in range(K_init):
                estimates = self._sample_estimates(expected_param_inv)
                init_mean[k,n], init_var[k,n] = estimates
                # leave the entropy, as we refrain from estimating it

        self.mean = init_mean
        self.var = init_var
        self.etrpy = init_etrpy
        self.mean_sq = mean_sq(self.mean, self.var)

    def _sample_estimates(self, expected_var):
        from npbNMF.datatools import sample_init_factor_TN
        estimates  = sample_init_factor_TN(self.lower_bound, self.upper_bound,
                                           np.sqrt(expected_var), n_samples=10)
        return estimates

    def update(self, X, mean_H, mean_sq_H, Z, mean_hyperpriors,
               mean_noise, feature_update_list, n=None):

        K = mean_H.shape[1]
        if mean_hyperpriors.ndim <= 1:
            # mean_hyperpriors has shape (K)
            mean_hyperpriors = mean_hyperpriors.reshape(K, 1)
            mean_hyperpriors = np.broadcast_to(mean_hyperpriors, (K, self.N))

        mean_sq_sum_H = np.sum(mean_sq_H, axis=0)

        if n:
            mask = np.asarray([True]*K)
            for k in feature_update_list:
                if Z[k,n]==0:
                    mu = 0
                    sigma_sq = 1/mean_hyperpriors[k,n]
                else:
                    mask[k] = False
                    sigma_sq = 1/(mean_noise*mean_sq_sum_H[k]
                                  + mean_hyperpriors[k,n])
                    prod_mix = mean_H[:,mask] @ (self.mean[mask,n]*Z[mask,n])
                    x_tilde = X[:,n] - prod_mix
                    hx_tilde = mean_H[:,k].T @ x_tilde
                    mu = sigma_sq*mean_noise*hx_tilde
                    mask[k] = True
                mean, var, etrpy, m_sq = self.moments(mu, sigma_sq)
                self.mean[k,n] = mean
                self.var[k,n] = var
                self.etrpy[k,n] = etrpy
                self.mean_sq[k,n] = m_sq
        else:
            # there is one sigma_sq for each W_nk
            sigma_sq = np.reciprocal(mean_noise*mean_sq_sum_H.reshape(-1,1)*Z
                                          + mean_hyperpriors) # note Z**2=Z
            HH_zero_diag = mean_H.T @ mean_H
            np.fill_diagonal(HH_zero_diag, 0)
            ZHX = Z*(mean_H.T @ X)
            for k in feature_update_list:
                prod_mix = ZHX[k,:] - Z[k,:]*(HH_zero_diag[k,:]
                                                    @ (Z*self.mean))
                mu =  sigma_sq[k,:]*(mean_noise * prod_mix)
                mean, var, etrpy, m_sq = self.moments(mu, sigma_sq[k,:])
                self.mean[k,:] = mean
                self.var[k,:] = var
                self.etrpy[k,:] = etrpy
                self.mean_sq[k,:] = m_sq

    def elbo_part(self, mean_hyperprior, ln_mean_hyperprior):
        """ Calculate ELBO contribution

            PRIOR ON W: W ~ TruncatedNormal(mu=0, var=hyperprior,
                                            lower_bound=0,upper_bound=infity)

            This is a truncated normal, thus we need to account for
            Z = cdf(inf) - cdf(0) = 1/2. As p(W|lambda)=Norm(0,lambda)/Z
            Further, we use that -ln(Z) = ln(2).

        INPUT
        =====
            - mean_param: array-like, shape (K) or (K,D)
            - ln_mean_param: array-like, shape (K) or (K,D)

        OUTPUT
        ======
            - elbo_part: float

        """
        # we utilize broadcasting (but it requires a transpositions)
        # this is safe, also if hyperpriors has shape (K)
        transposed_tmp = ln_mean_hyperprior.T\
                         - self.mean_sq.T*mean_hyperprior.T
        prior_elbo = np.sum(-0.5*np.log(2*np.pi) + 0.5*transposed_tmp
                            + np.log(2))
        entropy_elbo = np.sum(self.etrpy)
        return prior_elbo + entropy_elbo

    def add_new_features(self, expected_param_inv, k_new):
        new_moments = self._initialize_new_features(expected_param_inv, k_new)
        new_mean, new_var, new_etrpy = new_moments
        new_mean_sq = mean_sq(new_mean, new_var)
        self.mean = np.append(self.mean, new_mean, axis=0)
        self.var = np.append(self.var, new_var, axis=0)
        self.mean_sq = np.append(self.mean_sq, new_mean_sq, axis=0)
        self.etrpy = np.append(self.etrpy, new_etrpy, axis=0)

    def _initialize_new_features(self, expected_param_inv, k_new):
        new_mu = np.zeros((k_new, self.N))# the prior
        sigma_sq_expected = np.full((k_new, self.N), expected_param_inv)
        new_mean = np.empty((k_new, self.N))
        new_var = new_mean.copy()
        new_etrpy = new_var.copy()
        for k in range(k_new):
            new_mean[k,:], new_var[k,:], new_etrpy[k,:] = trunc_moments(
                                                         self.lower_bound,
                                                         self.upper_bound,
                                                         new_mu[k,:],
                                                         sigma_sq_expected[k,:]
                                                                      )

        return new_mean, new_var, new_etrpy

    def prune_features(self, mask):
        self.mean = self.mean[mask,:]
        self.var = self.var[mask,:]
        self.etrpy = self.etrpy[mask,:]
        self.mean_sq = self.mean_sq[mask,:]

    def get_attributes(self):
        return (self.mean.copy(), self.var.copy(), self.mean_sq.copy(),
                self.etrpy.copy())

    def set_attributes(self, attributes):
        self.mean, self.var, self.mean_sq, self.etrpy = attributes

##############################################################################

# prior objects for the model

##############################################################################

class NoiseGamma:

    def __init__(self,X, noise_alpha, noise_beta, data_size):
        self.noise_alpha = noise_alpha
        self.noise_beta = noise_beta
        self.X = X
        self.alpha = noise_alpha + 0.5*data_size

    def update(self, mean_H, mean_sq_H, mean_WZ,
               mean_sq_WZ):
        mean_sq_sum_H = np.sum(mean_sq_H, axis=0)
        mean_sq_sum_WZ = np.sum(mean_sq_WZ, axis=1)
        sum_mean_sq_error =  mean_X_LR_error_fast(self.X, mean_H,
                                                  mean_sq_sum_H, mean_WZ,
                                                  mean_sq_sum_WZ)
        # update beta (alpha needs only one update - see __init__())
        self.beta = self.noise_beta + 0.5*sum_mean_sq_error

        self.moments()

    def elbo_part(self):
        prior_elbo =  gamma_prior_elbo(self.mean, self.ln_mean,
                                       self.noise_alpha, self.noise_beta)
        entropy_elbo = self.etrpy

        return prior_elbo + entropy_elbo

    def moments(self):
        moments = gamma_moments(self.alpha, self.beta)
        self.mean, self.var, self.ln_mean, self.etrpy = moments

class BaseHyperprior:
    """
        HYPERPRIOR FOR lambda: lambda ~ Gamma(alpha, beta)
    """
    def __init__(self, alpha, beta):
        """ Initialize hyperpriors for the factors H and W

            INPUTS:
                    - alpha: float
                    - beta: float

        """
        self.hyperprior_alpha = alpha
        self.hyperprior_beta = beta

    def elbo_part(self):
        prior_elbo =  np.sum(gamma_prior_elbo(self.mean, self.ln_mean,
                             self.hyperprior_alpha, self.hyperprior_beta))

        entropy_elbo = np.sum(self.etrpy)

        return prior_elbo + entropy_elbo

    def moments(self, alpha, beta):
        moments = gamma_moments(alpha, beta)
        self.mean, self.var, self.ln_mean, self.etrpy = moments

    def get_attributes(self):
        return (self.mean.copy(), self.var.copy(), self.ln_mean.copy(),
                self.etrpy.copy())

    def set_attributes(self, attributes):
        self.mean, self.var, self.ln_mean, self.etrpy = attributes

class BaseSharedHyperprior(BaseHyperprior):

    def __init__(self, alpha, beta, K_init):
        super().__init__(alpha,beta)
        alpha = np.full(K_init, alpha)
        beta = np.full(K_init, beta)
        self.moments(alpha, beta)

    def add_new_features(self, k_new):
        new_moments = self.initialize_new_features(k_new)
        new_mean, new_var, new_ln_mean, new_etrpy = new_moments
        self.mean = np.append(self.mean, new_mean)
        self.var = np.append(self.var, new_var)
        self.ln_mean = np.append(self.ln_mean, new_ln_mean)
        self.etrpy = np.append(self.etrpy, new_etrpy)

    def initialize_new_features(self, k_new):
        alpha = np.full(k_new, self.hyperprior_alpha)
        beta = np.full(k_new, self.hyperprior_beta)
        moments = gamma_moments(alpha, beta)
        new_mean, new_var, new_ln_mean, new_etrpy = moments
        return new_mean, new_var, new_ln_mean, new_etrpy

    def prune_features(self, mask):
        self.mean = self.mean[mask]
        self.var = self.var[mask]
        self.etrpy = self.etrpy[mask]
        self.ln_mean = self.ln_mean[mask]

class HyperpriorH(BaseSharedHyperprior):
    """ Class governing hyperprior for factor H alone.

            Model places a hyperprior on each feature k, which is shared across
            D observed features. This hyperprior is NOT shared with W.
    """
    def __init__(self, alpha, beta, D, K_init):
        super().__init__(alpha, beta, K_init)
        self.alpha = alpha + 0.5*D

    def update(self, mean_sq_H):
        self.beta = self.hyperprior_beta + 0.5*np.sum(mean_sq_H, axis=0)
        self.moments(self.alpha, self.beta)

class HyperpriorShared(BaseSharedHyperprior):
    """ Class governing hyperprior for factor H and W.

            Model places a hyperprior on each feature k, and jointly shared
            across D observed features and N observations. This hyperprior IS
            shared with W.

            PRIOR ON W: W ~ TruncatedNormal(mu=0,var=hyperprior,
                                            lower_bound=0,upper_bound=infity)

    """
    def __init__(self, alpha, beta, D, N, K_init):
        super().__init__(alpha, beta, K_init)
        self.alpha = alpha + 0.5*(D + N)

    def update(self, mean_sq_H, mean_sq_W):
        self.beta = self.hyperprior_beta + 0.5*(np.sum(mean_sq_H, axis=0)
                                                + np.sum(mean_sq_W, axis=1))
        self.moments(self.alpha, self.beta)

class HyperpriorSharedWithSparse(BaseSharedHyperprior):
    """ Class governing hyperprior for factor W alone.

            Model places a hyperprior on each element in W - thus being
            sparsity-promoting.

            PRIOR ON W: W ~ TruncatedNormal(mu=0, var=hyperprior,
                                            lower_bound=0,upper_bound=infity)

    """
    def __init__(self, alpha, beta, D, N, K_init):
        super().__init__(alpha, beta, K_init)
        self.moments(np.full(K_init, alpha), np.full(K_init, beta))
        self.alpha = alpha + 0.5*(D + N)

    def update(self, mean_sq_H, mean_sq_W, mean_sparse_lambda):

        self.beta = self.hyperprior_beta + 0.5*(np.sum(mean_sq_H, axis=0)
                                         + np.sum(mean_sq_W*mean_sparse_lambda,
                                                axis=1))
        self.moments(self.alpha, self.beta)

class BaseSparseHyperprior(BaseHyperprior):

    def __init__(self, alpha, beta, N, K_init):
        super().__init__(alpha,beta)
        self.alpha = alpha + 0.5
        self.N = N
        alpha = np.full((K_init, N), alpha)
        beta = np.full((K_init, N), beta)
        self.moments(alpha, beta)

    def add_new_features(self, k_new):
        new_moments = self.initialize_new_features(k_new)
        new_mean, new_var, new_ln_mean, new_etrpy = new_moments
        self.mean = np.append(self.mean, new_mean, axis=0)
        self.var = np.append(self.var, new_var, axis=0)
        self.ln_mean = np.append(self.ln_mean, new_ln_mean, axis=0)
        self.etrpy = np.append(self.etrpy, new_etrpy, axis=0)

    def initialize_new_features(self, k_new):
        alpha = np.full((k_new,self.N), self.hyperprior_alpha)
        beta = np.full((k_new,self.N), self.hyperprior_beta)
        moments = gamma_moments(alpha, beta)
        new_mean, new_var, new_ln_mean, new_etrpy = moments
        return new_mean, new_var, new_ln_mean, new_etrpy

    def prune_features(self, mask):
        self.mean = self.mean[mask,:]
        self.var = self.var[mask,:]
        self.etrpy = self.etrpy[mask,:]
        self.ln_mean = self.ln_mean[mask,:]

class HyperpriorSparse(BaseSparseHyperprior):

    def __init__(self, alpha, beta, N, K_init):
        super().__init__(alpha, beta, N, K_init)

    def update(self, mean_sq_W):
        self.beta = self.hyperprior_beta + (0.5*mean_sq_W)
        self.moments(self.alpha, self.beta)

class HyperpriorSparseWithShared(BaseSparseHyperprior):
    """ Class governing hyperprior for factor W alone.

            Model places a hyperprior on each element in W - thus being
            sparsity-promoting.

            ADDITIONAL CONDITION: a hyperprior, shared between W and H, and
                                  placed on each feature is required.

            PRIOR ON W: W ~ TruncatedNormal(mu=0,
                                            var=hyperprior*shared_hyperprior,
                                            lower_bound=0,upper_bound=infity)

    """
    def __init__(self, alpha, beta, N, K_init):
        super().__init__(alpha, beta, N, K_init)

    def update(self, mean_sq_W, mean_lambda_shared):
        K = mean_sq_W.shape[0]
        self.beta = self.hyperprior_beta + 0.5*(mean_sq_W
                                         *mean_lambda_shared.reshape(K,1))
        self.moments(self.alpha, self.beta)
