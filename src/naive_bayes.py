import numpy as np
#from scipy.misc import logsumexp
from abc import ABC, abstractmethod

class BaseNaiveBayes(ABC):

    def predict(self, X):
        ll = self._log_likelihood(X)
        classes = self.classes[np.argmax(ll, axis=1)]
        return classes

    def predict_log_proba(self, X):
        ll = self._log_likelihood(X)
        Z = logsumexp(ll, axis=1)
        #Z = logsumexp(ll, axis=1, keepdims=True)
        return ll - Z

    def predict_proba(self, X):
        log_proba = self.predict_log_proba(X)
        proba = np.exp(log_proba)
        return proba

    @abstractmethod
    def _log_likelihood(self, X):
        pass

class GaussianNaiveBayes(BaseNaiveBayes):

    def __init__(self, class_prior=None):
        self.class_prior = class_prior

    def fit(self, X, y, weights=None):
        self.classes = np.sort(np.unique(y))
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.class_counts = np.zeros(self.n_classes)
        self.Mu_ = np.zeros((self.n_classes, self.n_features))
        self.Sigma_ = np.zeros((self.n_classes, self.n_features))

        for c in range(self.n_classes):
            Xc = X[y == c, :]
            if weights is not None:
                Wc = weights[y == c]
                Nc = Wc.sum()
            else:
                Wc = None
                Nc = Xc.shape[0]
            self.class_counts[c] = Nc
            mu_c, Sigma_c = self._update_mean_variance(Xc, Wc)
            self.Mu_[c, :] = mu_c
            self.Sigma_[c, :] = Sigma_c

        if self.class_prior is not None:
            self.class_probs = self.class_prior / self.class_prior.sum()
        else:
            self.class_probs = self.class_counts / self.class_counts.sum()

    def _log_likelihood(self, X):
        N = X.shape[0]
        log_likelihood = np.zeros((N, self.n_classes))
        for c in range(self.n_classes):
            n_ij = self._log_normal_probs(X, self.Mu_[c, :], self.Sigma_[c, :])
            log_likelihood[:, c] = np.sum(n_ij, axis=1) + np.log(self.class_probs[c])
        return log_likelihood

    def _update_mean_variance(self, X, weights):
        if weights is not None:
            norm = weights.sum()
            mu = np.average(X, axis=0, weights=weights / norm)
            Sigma = np.average((X - mu)**2, axis=0, weights=weights / norm)
        else:
            mu = np.mean(X, axis=0)
            Sigma = np.var(X, axis=0)
        return mu, Sigma

    def _log_normal_probs(self, X, mu, Sigma):
        n_ij = -0.5 * np.log(2.0 * np.pi * Sigma) - 0.5 * (X - mu)**2 / Sigma
        return n_ij


class BaseDiscreteNaiveBayes(BaseNaiveBayes):

    def _update_class_log_prob(self, class_prior):
        n_classes = self.n_classes
        if class_prior is not None:
            self.class_log_prob = np.log(class_prior)
        else:
            if self.fit_prior:
                self.class_log_prob = np.log(self.class_counts) - np.log(self.class_counts.sum())
            else:
                self.class_log_prob = np.full(n_classes, -np.log(n_classes))

    def _encode_target(self, y_train):
        N = len(y_train)
        self.classes = np.sort(np.unique(y_train))
        self.n_classes = len(self.classes )
        Y = np.zeros((N, self.n_classes))
        Y[np.arange(N), y_train] = 1.0
        return Y

    def fit(self, X, y):
        Y = self._encode_target(y)
        self._count(X, Y)
        self._update_class_log_prob(self.class_prior)
        self._update_feature_log_prob(self.alpha)

    @abstractmethod
    def _count(self, X, Y):
        pass

    @abstractmethod
    def _update_feature_log_prob(self, alpha):
        pass


class MultinomialNaiveBayes(BaseDiscreteNaiveBayes):

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _count(self, X, Y):
        self.feature_counts = np.dot(Y.T, X)
        self.class_counts = np.sum(Y, axis=0)

    def _update_feature_log_prob(self, alpha):
        Njc = self.feature_counts + alpha
        Nc = np.sum(Njc, axis=1, keepdims=True)
        self.feature_log_prob = np.log(Njc) - np.log(Nc)

    def _log_likelihood(self, X):
        ll = np.dot(X, self.feature_log_prob.T) + self.class_log_prob
        return ll


def logsumexp(X, axis):
    max_val = np.max(X, axis=axis, keepdims=True)
    res = np.log(np.sum(np.exp(X - max_val), axis=axis, keepdims=True)) + max_val
    return res