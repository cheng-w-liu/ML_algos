import numpy as np
from scipy.misc import logsumexp
from abc import ABC, abstractmethod

class BaseNaiveBayes(ABC):

    def predict(self, X):
        ll = self._log_likelihood(X)
        classes = self.classes[np.argmax(ll, axis=1)]
        return classes

    def predict_log_proba(self, X):
        ll = self._log_likelihood(X)
        Z = logsumexp(ll, axis=1, keepdims=True)
        return ll - Z

    def predict_proba(self, X):
        log_proba = self.predict_log_proba(X)
        proba = np.exp(log_proba)
        return proba

    @abstractmethod
    def _log_likelihood(self, X):
        pass


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
