import sys
import time
import os
import argparse
from pathlib import Path
from sacred import Experiment

from data_loader import data_ingredient, data_loader

# create experiment, using the data_loader ingredient
ex = Experiment('NMF', ingredients=[data_ingredient])

@ex.named_config
def malNMF():
    model_type = "malNMF"
    W_hyperprior = "NOT RELEVANT - THIS IS MAL"
    K = 25

@ex.named_config
def malPCA():
    model_type = "malPCA"
    W_hyperprior = "NOT RELEVANT - THIS IS MAL"

@ex.named_config
def gibbsvb():
    model_type = "gibbsvb"
    W_hyperprior = "shared"
    K_init = 1
    tolerance = 1e-4
    max_iter = 1000
    num_samples = 500

@ex.named_config
def gibbs():
    model_type = "gibbs"
    W_hyperprior = "shared"
    alpha_beta_prior = False
    K_init = 1
    num_samples = 1000

@ex.named_config
def vb():
    model_type = "vb"
    W_hyperprior = "shared_sparse"
    prior_type = "trunc_norm"
    K = 25
    tolerance = 1e-5
    max_iter = 1000

@ex.named_config
def vb_ibp():
    model_type = "vb_ibp"
    W_hyperprior = "shared_sparse"
    prior_type = "trunc_norm"
    K = 25
    tolerance = 1e-5
    max_iter = 1000
    pi_beta = 1e-0
    pi_alpha = 5e-0*pi_beta

@ex.config
def cfg():
    # Noise prior, Gamma distribution
    a = 1e-0 # For noise
    b = 1e-1 # For noise
    # Factor hyperpriors, both Gamma distributions
    c = 1e-0 # For H
    d = 1e-1 # For H
    e = 1e-0 # For W
    f = 1e-1 # For W
    # Alpha, Beta hyperpriors
    g = 5e-0 # For alpha
    l = 1e-0 # For alpha
    m = 1e-0 # For beta
    o = 1e-0 # For beta
    K = 25 # For VB methods

    alpha = 5 # For GibbsVB
    beta = 1 # For GibbsVB
    experiment_path = None


@ex.named_config
def basic():
    experiment_path = Path(__file__).parent / "results" / "raw_data_basic"

@ex.named_config
def learning_curve():
    experiment_path = Path(__file__).parent / "results" / ("raw_data_"
    + "learning_curve")

@ex.named_config
def EEG():
    experiment_path = Path(__file__).parent / "results" / "classifiers" /\
                      "learned_representations"

@ex.named_config
def uninformative():
    experiment_path = Path(__file__).parent / "results" / ("MCMC_uninformative"
    + "_prior")
    a = 1e-0 # For noise
    b = 1e-0 # For noise
    c = 1e-0
    d = 1e+2
    e = 1e-0
    f = 1e+2
    g = 4.0 # For alpha for Gibbs
    l = 1.0 # For alpha for Gibbs
    m = 0.5 # For beta for Gibbs
    o = 1.0 # For beta for Gibbs
    K_init = 25
    alpha = 4.0 # For GibbsVB
    beta = 0.5 # For GibbsVB

@ex.capture
def select_gibbs_vb(X, K_init, tolerance, max_iter, a, b, c, d, e, f,
                    alpha, beta, num_samples, W_hyperprior):
    if W_hyperprior == "shared":
        from npbNMF.vb_gibbs_inference import GibbsVBSharedNPB
        return GibbsVBSharedNPB(X, K_init, tolerance, max_iter, a, b, c, d,
                                alpha, beta, num_samples)
    elif W_hyperprior == "sparse":
        from npbNMF.vb_gibbs_inference import GibbsVBSparseNPB
        return GibbsVBSparseNPB(X, K_init, tolerance, max_iter, a, b, c, d, e,
                                f, alpha, beta, num_samples)
    elif W_hyperprior == "shared_sparse":
        from npbNMF.vb_gibbs_inference import GibbsVBSharedSparseNPB
        return GibbsVBSharedSparseNPB(X, K_init, tolerance, max_iter, a, b, c,
                                      d, e, f, alpha, beta, num_samples)
    else:
        sys.exit("Unkown hyperprior")

@ex.capture
def select_gibbs(X, K_init, a, b, c, d, e, f, g, l, m, o, num_samples,
                 alpha_beta_prior, W_hyperprior):
    if W_hyperprior == "shared":
        from npbNMF.gibbs_inference import GibbsSharedNPB
        return GibbsSharedNPB(X, K_init, a, b, c, d, g, l, m, o, num_samples,
                              alpha_beta_prior)
    elif W_hyperprior == "sparse":
        from npbNMF.gibbs_inference import GibbsSparseNPB
        return GibbsSparseNPB(X, K_init, a, b, c, d, e, f, g, l, m, o,
                              num_samples, alpha_beta_prior)
    elif W_hyperprior == "shared_sparse":
        from npbNMF.gibbs_inference import GibbsSharedSparseNPB
        return GibbsSharedSparseNPB(X, K_init, a, b, c, d, e, f, g, l, m, o,
                                    num_samples, alpha_beta_prior)
    else:
        sys.exit("Unkown hyperprior")

@ex.capture
def select_vb(X, K, tolerance, max_iter, a, b, c, d, e, f, W_hyperprior,
              prior_type):
    if W_hyperprior == "shared":
        from npbNMF.vb_inference import NMFSharedHyperprior
        return NMFSharedHyperprior(X, K, tolerance, max_iter, a, b, c, d,
                  prior_type)
    elif W_hyperprior == "sparse":
        from npbNMF.vb_inference import NMFDoubleHyperprior
        return NMFDoubleHyperprior(X, K, tolerance, max_iter, a, b, c, d, e, f,
                                   prior_type)
    elif W_hyperprior == "shared_sparse":
        from npbNMF.vb_inference import NMFDoubleHyperpriorWithShared
        return NMFDoubleHyperpriorWithShared(X, K, tolerance, max_iter, a, b,
                                             c, d, e, f, prior_type)
    else:
        sys.exit("Unkown hyperprior")


@ex.capture
def select_vb_ibp(X, K, tolerance, max_iter, a, b, c, d, e, f, pi_alpha,
                  pi_beta, W_hyperprior, prior_type):
    if W_hyperprior == "shared":
        from npbNMF.vb_inference import NPBNMFSharedHyperprior
        return NPBNMFSharedHyperprior(X, K, tolerance, max_iter, a, b, c, d,
                                      pi_alpha, pi_beta, prior_type)
    elif W_hyperprior == "sparse":
        from npbNMF.vb_inference import NPBNMFDoubleHyperprior
        return NPBNMFDoubleHyperprior(X, K, tolerance, max_iter, a, b, c, d, e,
                                      f, pi_alpha, pi_beta, prior_type)
    elif W_hyperprior == "shared_sparse":
        from npbNMF.vb_inference import NPBNMFDoubleHyperpriorWithShared
        return NPBNMFDoubleHyperpriorWithShared(X, K, tolerance, max_iter, a,
                                                b, c, d, e, f, pi_alpha,
                                                pi_beta, prior_type)
    else:
        sys.exit("Unkown hyperprior")

@ex.capture
def select_malNMF(X, K):
    from mal_decomposition import NMFmal
    return NMFmal(X, K)

@ex.capture
def select_malPCA(X):
    from mal_decomposition import PCAmal
    return PCAmal(X)

@ex.capture
def get_random_seed(_seed):
    """
    Everytime this function is called a new _seed is generated based on the
    global seed in sacred.

    """
    return _seed

@ex.capture
def model_selector(model_type, X):

    if model_type == "vb":
        return select_vb(X)
    elif model_type == "vb_ibp":
        return select_vb_ibp(X)
    elif model_type == "gibbs":
        return select_gibbs(X)
    elif model_type == "gibbsvb":
        return select_gibbs_vb(X)
    elif model_type == "malNMF":
        return select_malNMF(X)
    elif model_type == "malPCA":
        return select_malPCA(X)

@ex.automain
def main(dataset, model_type, experiment_path, W_hyperprior, _run, _log):

    classes = dataset['classes']
    subject = dataset['subject']
    if not classes:
        X = data_loader()
        _, N = X.shape
    else:
        import numpy as np
        Xtrain, Xtest, Ytrain, Ytest = data_loader()
        _, divisor = Xtrain.shape
        X = np.append(Xtrain, Xtest, axis=1)
        D, N = X.shape
    msg =(f"\n\t-MODEL TYPE: {model_type}\n\t-W PRIOR TYPE: {W_hyperprior}"
          +f"\n\t-Data Size: {N}")
    if classes:
        msg += f"\n\nRUNNING EEG\n\n\t -Subject: {subject}"
    _log.info(msg)
    model = model_selector(X=X) # all other arguments set by sacred

    timer = time.time()
    model.train(_run.info, _log)
    timer = time.time() - timer
    _run.info['training_time'] = timer


    if experiment_path and not classes:
        if "uninform" in str(experiment_path):
            experiment_path = experiment_path.resolve() / model_type / str(N)
        else:
            experiment_path = experiment_path.resolve() / model_type /\
                                                          W_hyperprior

        filename = experiment_path / "factors.pickle"
        model.pickle_factors(filename) # store model configuration
        _run.add_artifact(filename, name="model_configuration_" + model_type
                          + ".pickle") # add the generated
                                       # pickle of factors to the experiment
    if experiment_path and classes:
        experiment_path = experiment_path.resolve()
        filename = experiment_path / f"factors.pickle"
        model.pickle_factors(filename) # store model configuration
        _run.add_artifact(filename, name="model_configuration_" + model_type
                          + ".pickle") # add the generated
                                       # pickle of factors to the experiment

