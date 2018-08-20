from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import scale
import seaborn as sns
import matplotlib.pyplot as plt

class NMFmal(NMF):

    def __init__(self, X, K):
        super().__init__(n_components=K, init="nndsvd", solver='cd')
        self.X = X.T

    def train(self, info_dict, log):
        log.info("Training maximum likelihood: NMF")
        self.W = self.fit_transform(self.X).T
        self.H = self.components_.T

    def pickle_factors(self, filename):
        import pickle
        feature_dict = {'H': self.H,
                        'W': self.W}
        with open(filename, 'bw') as f:
            pickle.dump(feature_dict, f)

class PCAmal(PCA):

    def __init__(self, X):
        super().__init__(svd_solver="full")
        self.X = X.T
        self.X = scale(self.X)

    def train(self, info_dict, log):
        log.info("Training maximum likelihood: PCA")
        self.W = self.fit_transform(self.X)

    def pickle_factors(self, filename):
        import pickle
        feature_dict = {'W': self.W.T,
                        'var_explained': self.explained_variance_,
                        'var_explained_ratio': self.explained_variance_ratio_}
        with open(filename, 'bw') as f:
            pickle.dump(feature_dict, f)
