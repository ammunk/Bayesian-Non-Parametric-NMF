import numpy as np
from scipy.special import expit, gammaln
from npbNMF.datatools import trunc_moments, means_factor_prod,\
                             gamma_moments, gamma_prior_elbo, \
                             beta_moments, bernoulli_moments, \
                             expon_moments, mean_sq, mean_X_LR_error_fast

##############################################################################

# factor objects for the model

##############################################################################

class FactorTruncNormVB:

    def __init__(self, lower_bound, upper_bound, K):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.K = K
        self.etrpy = None

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


class BaseH(FactorTruncNormVB):

    def __init__(self, lower_bound, upper_bound, K, D):
        super().__init__(lower_bound, upper_bound, K)
        self.D = D
        self.mean = np.empty([D, K])
        self.mean_sq = np.empty_like(self.mean)
        self.var = np.empty_like(self.mean)
        self.etrpy = np.empty_like(self.mean)

    def initialize(self, expected_param):
        init_mean =  np.empty([self.D, self.K])
        init_var = np.empty_like(init_mean)
        init_etrpy = np.empty_like(init_mean)

        from npbNMF.datatools import sample_init_factor_expon
        for d in range(self.D):
            for k in range(self.K):
                estimates  = sample_init_factor_expon(expected_param,
                                                      n_samples=10)
                init_mean[d,k], init_var[d,k] = estimates
                # leave the entropy, as we refrain from estimating it

        self.mean = init_mean
        self.var = init_var
        self.etrpy = init_etrpy
        self.mean_sq = mean_sq(self.mean, self.var)

    def update(self, X, mean_R, mean_sq_R, mean_hyperpriors, mean_noise):
        mean_sq_sum_R = np.sum(mean_sq_R, axis=1)
        noise_R_sq_sum = mean_noise*mean_sq_sum_R

        RR_zero_diag = mean_R @ mean_R.T
        np.fill_diagonal(RR_zero_diag, 0)
        RX = mean_R @ X.T

        for k in np.random.permutation(self.K):
            prod_mix = RX[k,:] -  RR_zero_diag[k,:] @ self.mean.T
            # mu shape (D), sigma_sq shape (1)
            mu, sigma_sq = self._calc_mu_sigma(mean_noise,
                                               mean_hyperpriors[k],
                                               noise_R_sq_sum[k],
                                               prod_mix)
            moments = self.moments(mu, sigma_sq)
            new_mean, new_var, new_etrpy, new_mean_sq = moments
            self.mean[:,k] = new_mean
            self.var[:,k] = new_var
            self.etrpy[:,k] = new_etrpy
            self.mean_sq[:,k] = new_mean_sq

    def _calc_mu_sigma(self, mean_noise, mean_hyperpriors, noise_R_sq_sum,
                       prod_mix):
        pass


    def moments(self, mu, sigma_sq):
        new_mean, new_var,  new_etrpy = trunc_moments(
                                                      self.lower_bound,
                                                      self.upper_bound,
                                                      mu,
                                                      sigma_sq)
        new_mean_sq = mean_sq(new_mean, new_var)
        return new_mean, new_var, new_etrpy, new_mean_sq

class FactorHTruncNorm(BaseH):

    def __init__(self, lower_bound, upper_bound, K, D):
        super().__init__(lower_bound, upper_bound, K, D)

    def _calc_mu_sigma(self, mean_noise, mean_hyperpriors, noise_R_sq_sum,
                       prod_mix):
        sigma_sq = np.reciprocal(noise_R_sq_sum + mean_hyperpriors)
        mu =  sigma_sq*mean_noise*prod_mix
        return mu, sigma_sq

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

class FactorHExpon(BaseH):

    def __init__(self, lower_bound, upper_bound, K, D):
        super().__init__(lower_bound, upper_bound, K, D)

    def _calc_mu_sigma(self, mean_noise, mean_hyperpriors, noise_R_sq_sum,
                       prod_mix):
        sigma_sq = np.reciprocal(noise_R_sq_sum)
        if sigma_sq<0:
            import sys
            sys.exit(("Vanishing features, which cannot be handled with an"
                      +"exponential prior \n\t- Try with a Truncated Normal"
                      + "prior"))
        mu =  sigma_sq*(mean_noise*prod_mix - mean_hyperpriors)
        return mu, sigma_sq

    def elbo_part(self, mean_hyperprior, ln_mean_hyperprior):
        """ Calculate ELBO contribution

            PRIOR ON H: H ~ Exponential(hyperprior)

        INPUT
        =====
            - mean_param: array-like, shape (K)
            - ln_mean_param: array-like, shape (K)

        OUTPUT
        ======
            - elbo_part: float

        """

        prior_elbo = self.D*np.sum(ln_mean_hyperprior)
        prior_elbo -= np.sum(self.mean*mean_hyperprior) # utilize broadcasting
        entropy_elbo = np.sum(self.etrpy)

        return prior_elbo + entropy_elbo

class BaseW(FactorTruncNormVB):

    def __init__(self, lower_bound, upper_bound, K, N):
        super().__init__(lower_bound, upper_bound, K)
        self.N = N
        self.mean = np.empty([K,N])
        self.mean_sq = np.empty_like(self.mean)
        self.var = np.empty_like(self.mean)
        self.etrpy = np.empty_like(self.mean)

    def initialize(self, expected_param):
        init_mean =  np.empty([self.K, self.N])
        init_var = np.empty_like(init_mean)
        init_etrpy = np.empty_like(init_mean)

        from npbNMF.datatools import sample_init_factor_expon
        for n in range(self.N):
            for k in range(self.K):
                estimates = self._sample_estimates(expected_param)
                init_mean[k,n], init_var[k,n] = estimates
                # leave the entropy, as we refrain from estimating it

        self.mean = init_mean
        self.var = init_var
        self.etrpy = init_etrpy
        self.mean_sq = mean_sq(self.mean, self.var)

    def _sample_estimates(self, expected_param):
        pass


class BaseFactorW(BaseW):

    def __init__(self, lower_bound, upper_bound, K, N):
        super().__init__(lower_bound, upper_bound, K, N)

    def update(self, X, mean_H, mean_sq_H, mean_hyperpriors, mean_noise):
        """ Updates the mean, variance, entropy and mean_sq

            INPUTS:
            =======
                    - X: array-like, shape (D, N)
                    - mean_hyperpriors: array-like, shape (K) or (K,N)
                                        - (K) if mean_hyperprior for each K
                                          feature is shared by all observations
                                        - (K,N) if mean_hyperprior is uniquely
                                          associated with each r_kn
                                        - OR (K,N) if mean_hyperprior=a*b, with
                                          shape(a) = (K,N) and shape(b) = (K),
                                          s.t. a_kn is uniquely associated with
                                          each r_kn, but b_k is shared across
                                          observations
                    - mean_noise: float, expected precision associated with the
                                  likelihood
                    - mean_H: array-like, shape (D,K)
                    - mean_sq_H: array-like, shape(D,K)
        """

        if mean_hyperpriors.ndim <= 1:
            # mean_hyperpriors has shape (K)
            mean_hyperpriors = mean_hyperpriors.reshape(self.K, 1)

        mean_sq_sum_H = np.sum(mean_sq_H, axis=0) # shape (K)
        noise_H_sq_sum = mean_noise*mean_sq_sum_H # shape (K)

        HH_zero_diag = mean_H.transpose() @ mean_H
        np.fill_diagonal(HH_zero_diag, 0)
        HX = mean_H.T @ X

        for k in np.random.permutation(self.K):
            prod_mix =  HX[k,:] - HH_zero_diag[k,:] @ self.mean # shape(N)
            mu, sigma_sq = self._calc_mu_sigma(mean_noise,
                                               mean_hyperpriors[k,:],
                                               noise_H_sq_sum[k], prod_mix)
            moments = self.moments(mu, sigma_sq)
            new_mean, new_var, new_etrpy, new_mean_sq = moments
            self.mean[k,:] = new_mean
            self.var[k,:] = new_var
            self.etrpy[k,:] = new_etrpy
            self.mean_sq[k,:] = new_mean_sq

    def _calc_mu_sigma(self, mean_noise, mean_hyperpriors, noise_H_sq_sum,
                       prod_mix):
        pass

class BaseWExpon(BaseW):

    def __init__(self, lower_bound, upper_bound, K, N):
        super().__init__(lower_bound, upper_bound, K, N)

    def _calc_mu_sigma(self, mean_noise, mean_hyperpriors, noise_H_sq_sum,
                       prod_mix):
        """ Calculates mu and sigma squared for a k'th feature across all
            observations

            PRIOR ON W: W ~ Exponential(hyperprior)

        """
        sigma_sq = np.reciprocal(noise_H_sq_sum) # shape (N)
        mu =  sigma_sq*(mean_noise*prod_mix - mean_hyperpriors) # shape (N)
        return mu, sigma_sq

    def _sample_estimates(self, expected_param):
        from npbNMF.datatools import sample_init_factor_expon
        estimates  = sample_init_factor_expon(expected_param, n_samples=10)
        return estimates

    def elbo_part(self, mean_hyperprior, ln_mean_hyperprior):
        """ Calculate ELBO contribution

            PRIOR ON W: W ~ Exponential(hyperprior)

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
                         - self.mean.T*mean_hyperprior.T
        prior_elbo = np.sum(transposed_tmp) # as we sum everything, we
                                            # disregard a retransposition
        entropy_elbo = np.sum(self.etrpy)
        return prior_elbo + entropy_elbo


class BaseWTruncNorm(BaseW):

    def __init__(self, lower_bound, upper_bound, K, N):
        super().__init__(lower_bound, upper_bound, K, N)

    def _calc_mu_sigma(self, mean_noise, mean_hyperpriors, noise_H_sq_sum,
                       prod_mix):
        """ Calculates mu and sigma squared for a k'th feature across all
            observations

            PRIOR ON W: W ~ TruncatedNormal(mu=0, var=hyperprior,
                                            lower_bound=0,upper_bound=infity)

        """
        sigma_sq = np.reciprocal(noise_H_sq_sum + mean_hyperpriors) # shape (N)
        mu =  sigma_sq*(mean_noise*prod_mix) # shape(N)
        return mu, sigma_sq

    def _sample_estimates(self, expected_param):
        from npbNMF.datatools import sample_init_factor_TN
        estimates  = sample_init_factor_TN(self.lower_bound, self.upper_bound,
                                           1/np.sqrt(expected_param),
                                           n_samples=10)
        return estimates

    def elbo_part(self, mean_hyperprior, ln_mean_hyperprior):
        """ Calculate ELBO contribution

            PRIOR ON W: W ~ TruncatedNormal(mu=0, var=hyperprior,
                                            lower_bound=0,upper_bound=infity)

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

class FactorWExpon(BaseWExpon, BaseFactorW):

    def __init__(self, lower_bound, upper_bound, K, N): # BaseFactorW has to be last!!
        super().__init__(lower_bound, upper_bound, K, N)

class FactorWTruncNorm(BaseWTruncNorm, BaseFactorW): # BaseFactorW has to be last!!

    def __init__(self, lower_bound, upper_bound, K, N):
        super().__init__(lower_bound, upper_bound, K, N)

class BaseNPBW(BaseW):

    def __init__(self, lower_bound, upper_bound, K, N):
        super().__init__(lower_bound, upper_bound, K, N)

    def update(self, X, mean_H, mean_sq_H, mean_Z, mean_sq_Z, mean_hyperpriors,
               mean_noise):

        if mean_hyperpriors.ndim <= 1:
            # mean_hyperpriors has shape (K)
            mean_hyperpriors = mean_hyperpriors.reshape(self.K, 1)

        mean_sq_sum_H = np.sum(mean_sq_H, axis=0)
        noise_H_sq_sum_Z = mean_noise*mean_sq_sum_H.reshape(self.K,1)*mean_sq_Z

        HH_zero_diag = mean_H.T @ mean_H
        np.fill_diagonal(HH_zero_diag, 0)
        ZHX = mean_Z*(mean_H.T @ X)

        for k in np.random.permutation(self.K):
            prod_mix = ZHX[k,:] - mean_Z[k,:]*(HH_zero_diag[k,:]
                                               @ (mean_Z*self.mean))
            mu, sigma_sq = self._calc_mu_sigma(mean_noise,
                                               mean_hyperpriors[k,:],
                                               noise_H_sq_sum_Z[k,:], prod_mix)
            moments = self.moments(mu, sigma_sq)
            new_mean, new_var, new_etrpy, new_mean_sq = moments
            self.mean[k,:] = new_mean
            self.var[k,:] = new_var
            self.etrpy[k,:] = new_etrpy
            self.mean_sq[k,:] = new_mean_sq

    def _calc_mu_sigma(self, mean_noise, mean_hyperpriors, noise_H_sq_sum,
                       prod_mix):
        pass

class NPBFactorWExpon(BaseWExpon, BaseNPBW): # BaseNPBW HAS TO BE LAST!

    def __init__(self, lower_bound, upper_bound, K, N):
        super().__init__(lower_bound, upper_bound, K, N)

class NPBFactorWTruncNorm(BaseWTruncNorm, BaseNPBW): # BaseNPBW HAS TO BE LAST!

    def __init__(self, lower_bound, upper_bound, K, N):
        super().__init__(lower_bound, upper_bound, K, N)

class FactorBernoulli:

    def __init__(self, K, N):
        self.K = K
        self.N = N
        self.mean = np.empty([K,N])
        self.mean_sq = np.empty_like(self.mean)
        self.var = np.empty_like(self.mean)
        self.etrpy = np.empty_like(self.mean)

    def initialize_preset(self, init_mean, init_var):
        self.mean = init_mean
        self.var = init_var
        self.mean_sq = mean_sq(self.mean, self.var)

    def initialize(self, alpha, beta, sample_init=True):
        init_mean =  np.empty([self.K, self.N])
        init_var = np.empty_like(init_mean)
        init_etrpy = np.empty_like(init_mean)
        alpha = alpha/self.K
        beta = beta*(1 - 1.0/self.K)

        if sample_init:
            from npbNMF.datatools import sample_init_Z_bernoulli
            for n in range(self.N):
                for k in range(self.K):
                    estimates  = sample_init_Z_bernoulli(
                                                         alpha,
                                                         beta,
                                                         n_samples=10,
                                                         )
                    init_mean[k,n], init_var[k,n] = estimates
                    # leave the entropy, as we refrain from estimating it
        else:
            p_expected = np.full((self.K, self.N), alpha/(beta + alpha))
            init_mean, init_var, init_etrpy = bernoulli_moments(p_expected)

        self.mean = init_mean
        self.var = init_var
        self.etrpy = init_etrpy
        self.mean_sq = mean_sq(self.mean, self.var)

    def update(self, X, mean_H, mean_sq_H, mean_W, mean_sq_W, mean_ln_pi,
               mean_one_ln_pi, mean_noise):

        mean_sq_sum_H = np.sum(mean_sq_H, axis=0)
        HH_zero_diag = mean_H.T @ mean_H
        np.fill_diagonal(HH_zero_diag, 0)
        WHX = mean_W*(mean_H.T @ X)

        for k in np.random.permutation(self.K):
            prod_mix = WHX[k,:] - mean_W[k,:]*(HH_zero_diag[k,:]
                                                @ (mean_W*self.mean))
            pi =  mean_noise*(prod_mix - 0.5*mean_sq_W[k,:]*mean_sq_sum_H[k])
            pi += mean_ln_pi[k] - mean_one_ln_pi[k]
            pi = expit(pi) # sigmoid function

            moments = self.moments(pi)
            new_mean, new_var, new_etrpy, new_mean_sq = moments
            self.mean[k,:] = new_mean
            self.var[k,:] = new_var
            self.etrpy[k,:] = new_etrpy
            self.mean_sq[k,:] = new_mean_sq

    def elbo_part(self, ln_mean_pi, ln_o_mean_pi):
        # broadcasting will be done correctly
        prior_elbo = np.sum(self.mean*(ln_mean_pi - ln_o_mean_pi)
                            + ln_o_mean_pi)
        entropy_elbo = np.sum(self.etrpy)

        return prior_elbo + entropy_elbo


    def moments(self, pi):
        new_mean, new_var, new_etrpy = bernoulli_moments(pi)
        new_mean_sq = mean_sq(new_mean, new_var)
        return new_mean, new_var, new_etrpy, new_mean_sq

##############################################################################

# prior objects for the model

##############################################################################

class NoiseGamma:

    def __init__(self, X, noise_alpha, noise_beta, data_size):
        self.noise_alpha = noise_alpha
        self.noise_beta = noise_beta
        self.X = X
        self.alpha = noise_alpha + 0.5*data_size

    def update(self, mean_H, mean_sq_H, mean_R, mean_sq_R):
        mean_sq_sum_H = np.sum(mean_sq_H, axis=0)
        mean_sq_sum_R = np.sum(mean_sq_R, axis=1)

        sum_mean_sq_error =  mean_X_LR_error_fast(self.X, mean_H,
                                                  mean_sq_sum_H, mean_R,
                                                  mean_sq_sum_R)
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

class HyperpriorHExpon(BaseHyperprior):
    """ Class governing hyperprior for factor H alone.

            Model places a hyperprior on each feature k, which is shared across
            D observed features. This hyperprior is NOT shared with W.
    """
    def __init__(self, alpha, beta, D):
        super().__init__(alpha, beta)
        self.alpha = alpha + D

    def update(self, mean_H):
        self.beta = self.hyperprior_beta + np.sum(mean_H, axis=0)
        self.moments(self.alpha, self.beta)

class HyperpriorHTruncNorm(BaseHyperprior):
    """ Class governing hyperprior for factor H alone.

            Model places a hyperprior on each feature k, which is shared across
            D observed features. This hyperprior is NOT shared with W.
    """
    def __init__(self, alpha, beta, D):
        super().__init__(alpha, beta)
        self.alpha = alpha + 0.5*D

    def update(self, mean_sq_H):
        self.beta = self.hyperprior_beta + 0.5*np.sum(mean_sq_H, axis=0)
        self.moments(self.alpha, self.beta)

class HyperpriorExponShared(BaseHyperprior):
    """ Class governing hyperprior for factor H and W.

            Model places a hyperprior on each feature k, and jointly shared
            across D observed features and N observations. This hyperprior IS
            shared with W.

            PRIOR ON W: W ~ Exponential(hyperprior)

    """
    def __init__(self, alpha, beta, D, N):
        super().__init__(alpha, beta)
        self.alpha = alpha + D + N

    def update(self, mean_H, mean_W):
        self.beta = self.hyperprior_beta + (np.sum(mean_H, axis=0)
                                            + np.sum(mean_W, axis=1))
        self.moments(self.alpha, self.beta)

class HyperpriorTruncNormShared(BaseHyperprior):
    """ Class governing hyperprior for factor H and W.

            Model places a hyperprior on each feature k, and jointly shared
            across D observed features and N observations. This hyperprior IS
            shared with W.

            PRIOR ON W: W ~ TruncatedNormal(mu=0,var=hyperprior,
                                            lower_bound=0,upper_bound=infity)

    """
    def __init__(self, alpha, beta, D, N):
        super().__init__(alpha, beta)
        self.alpha = alpha + 0.5*(D + N)

    def update(self, mean_sq_H, mean_sq_W):
        self.beta = self.hyperprior_beta + 0.5*(np.sum(mean_sq_H, axis=0)
                                            + np.sum(mean_sq_W, axis=1))
        self.moments(self.alpha, self.beta)

class HyperpriorExponSharedWithSparsity(HyperpriorExponShared):
    """ Class governing hyperprior for factor W alone.

            Model places a hyperprior on each element in W - thus being
            sparsity-promoting.

            PRIOR ON W: W ~ Exponential(hyperprior)

    """
    def __init__(self, alpha, beta, D, N, K):
        super().__init__(alpha, beta, D, N)
        self.moments(np.full(K, self.alpha), np.full(K, beta))

    def update(self, mean_H, mean_W, mean_sparse_lambda):
        self.beta = self.hyperprior_beta + (np.sum(mean_H, axis=0)
                                            + np.sum(mean_W*mean_sparse_lambda,
                                                     axis=1))
        self.moments(self.alpha, self.beta)

class HyperpriorTruncNormSharedWithSparsity(HyperpriorTruncNormShared):
    """ Class governing hyperprior for factor W alone.

            Model places a hyperprior on each element in W - thus being
            sparsity-promoting.

            PRIOR ON W: W ~ TruncatedNormal(mu=0, var=hyperprior,
                                            lower_bound=0,upper_bound=infity)

    """
    def __init__(self, alpha, beta, D, N, K):
        super().__init__(alpha, beta, D, N)
        self.moments(np.full(K, alpha), np.full(K, beta))

    def update(self, mean_sq_H, mean_sq_W, mean_sparse_lambda):

        self.beta = self.hyperprior_beta + 0.5*(np.sum(mean_sq_H, axis=0)
                                     + np.sum(mean_sq_W*mean_sparse_lambda,
                                                  axis=1))
        self.moments(self.alpha, self.beta)

class HyperpriorTruncNormWSparsity(BaseHyperprior):

    def __init__(self, alpha, beta):
        super().__init__(alpha, beta)
        self.alpha = alpha + 0.5

    def update(self, mean_sq_W):
        self.beta = self.hyperprior_beta + 0.5*mean_sq_W

        self.moments(self.alpha, self.beta)

class HyperpriorExponWSparsity(BaseHyperprior):

    def __init__(self, alpha, beta):
        super().__init__(alpha, beta)
        self.alpha = alpha + 1

    def update(self, mean_W):
        self.beta = self.hyperprior_beta + mean_W

        self.moments(self.alpha, self.beta)

class HyperpriorTruncNormSparsityWithShared(HyperpriorTruncNormWSparsity):
    """ Class governing hyperprior for factor W alone.

            Model places a hyperprior on each element in W - thus being
            sparsity-promoting.

            ADDITIONAL CONDITION: a hyperprior, shared between W and H, and
                                  placed on each feature is required.

            PRIOR ON W: W ~ TruncatedNormal(mu=0,
                                            var=hyperprior*shared_hyperprior,
                                            lower_bound=0,upper_bound=infity)

    """
    def __init__(self, alpha, beta, N, K):
        super().__init__(alpha, beta)
        self.K = K
        self.moments(np.full((K, N), alpha),
                     np.full((K, N), beta))

    def update(self, mean_sq_W, mean_lambda_shared):
        self.beta = self.hyperprior_beta + 0.5*(mean_sq_W
                                         *mean_lambda_shared.reshape(self.K,1))
        self.moments(self.alpha, self.beta)

class HyperpriorExponSparsityWithShared(HyperpriorExponWSparsity):
    """ Class governing hyperprior for factor W alone.

            Model places a hyperprior on each element in W - thus being
            sparsity-promoting.

            ADDITIONAL CONDITION: a hyperprior, shared between W and H, and
                                  placed on each feature is required.

            PRIOR ON W: W ~ Exponential(hyperprior*shared_hyperprior)

    """
    def __init__(self, alpha, beta, N, K):
        super().__init__(alpha, beta)
        self.K = K
        self.moments(np.full((K, N), alpha),
                     np.full((K, N),beta))

    def update(self, mean_W, mean_lambda_shared):
        self.beta = self.hyperprior_beta + (mean_W
                                         *mean_lambda_shared.reshape(self.K,1))

        self.moments(self.alpha, self.beta)


class BetaPrior:

    def __init__(self, N, K, pi_alpha, pi_beta):
        self.N = N
        self.K = K

        # prior alpha and beta values
        self.prior_alpha = pi_alpha/self.K
        self.prior_beta = pi_beta*(1 - 1.0/self.K)
        self.ln_B = gammaln(self.prior_alpha) + gammaln(self.prior_beta)\
                    - gammaln(self.prior_alpha + self.prior_beta)

    def update(self, mean_Z):
        sum_Z = np.sum(mean_Z, axis=1).reshape(self.K, 1)
        self.alpha = self.prior_alpha + sum_Z
        self.beta = self.prior_beta + self.N - sum_Z

        self.moments()

    def elbo_part(self):
        prior_elbo = np.sum((self.prior_alpha - 1)*self.ln_mean
                            + (self.prior_beta - 1)*self.ln_o_mean - self.ln_B)
        entropy_elbo = np.sum(self.etrpy)

        return np.sum(prior_elbo) + np.sum(entropy_elbo)

    def moments(self):
        moments = beta_moments(self.alpha, self.beta)
        self.mean, self.var, self.ln_mean, self.ln_o_mean, self.etrpy = moments
