import warnings
from numbers import Integral, Real
import numpy as np
import scipy.linalg
from scipy import linalg

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import empirical_covariance, ledoit_wolf, shrunk_covariance
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

def get_namespace(*args, **kwargs):
    return np, False

def size(a):
    return np.size(a)

def _expit(x):
    from scipy.special import expit
    return expit(x)

def device(x):
    return None

def _cov(X, shrinkage=None, covariance_estimator=None):
    if covariance_estimator is None:
        shrinkage = "empirical" if shrinkage is None else shrinkage
        if isinstance(shrinkage, str):
            if shrinkage == "auto":
                sc = StandardScaler()
                X = sc.fit_transform(X)
                s = ledoit_wolf(X)[0]
                s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            elif shrinkage == "empirical":
                s = empirical_covariance(X)
        elif isinstance(shrinkage, Real):
            s = shrunk_covariance(empirical_covariance(X), shrinkage)
    else:
        if shrinkage is not None and shrinkage != 0:
            raise ValueError("covariance_estimator and shrinkage parameters are not None.")
        covariance_estimator.fit(X)
        s = covariance_estimator.covariance_
    return s

def _class_means(X, y):
    xp, is_array_api_compliant = get_namespace(X)
    classes, y = xp.unique_inverse(y)
    means = xp.zeros((classes.shape[0], X.shape[1]), device=device(X), dtype=X.dtype)
    cnt = np.bincount(y)
    np.add.at(means, y, X)
    means /= cnt[:, None]
    return means

def _class_cov(X, y, priors, shrinkage=None, covariance_estimator=None):
    classes = np.unique(y)
    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        cov += priors[idx] * np.atleast_2d(_cov(Xg, shrinkage, covariance_estimator))
    return cov

class LinearDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    def __init__(self, solver="svd", shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=1e-4, covariance_estimator=None):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol
        self.covariance_estimator = covariance_estimator

    def _solve_lstsq(self, X, y, shrinkage, covariance_estimator):
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_, shrinkage, covariance_estimator)
        self.coef_ = linalg.lstsq(self.covariance_, self.means_.T)[0].T
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_)

    def _solve_eigen(self, X, y, shrinkage, covariance_estimator):
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_, shrinkage, covariance_estimator)
        Sw = self.covariance_
        St = _cov(X, shrinkage, covariance_estimator)
        Sb = St - Sw
        evals, evecs = linalg.eigh(Sb, Sw)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][: self._max_components]
        evecs = evecs[:, np.argsort(evals)[::-1]]
        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_)

    def _solve_svd(self, X, y):
        xp, _ = get_namespace(X)
        svd = scipy.linalg.svd
        n_samples, n_features = X.shape
        n_classes = self.classes_.shape[0]
        self.means_ = _class_means(X, y)
        if self.store_covariance:
            self.covariance_ = _class_cov(X, y, self.priors_)
        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group]
            Xc.append(Xg - self.means_[idx, :])
        self.xbar_ = self.priors_ @ self.means_
        Xc = np.concatenate(Xc, axis=0)
        std = np.std(Xc, axis=0)
        std[std == 0] = 1.0
        fac = 1.0 / (n_samples - n_classes)
        X_scaled = np.sqrt(fac) * (Xc / std)
        U, S, Vt = svd(X_scaled, full_matrices=False)
        rank = np.sum(S > self.tol)
        scalings = (Vt[:rank, :] / std).T / S[:rank]
        fac = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)
        X_bet = ((np.sqrt((n_samples * self.priors_) * fac)) * (self.means_ - self.xbar_).T).T @ scalings
        _, S_bet, Vt_bet = svd(X_bet, full_matrices=False)
        self.explained_variance_ratio_ = (S_bet**2 / np.sum(S_bet**2))[: self._max_components]
        rank_bet = np.sum(S_bet > self.tol * S_bet[0])
        self.scalings_ = scalings @ Vt_bet.T[:, :rank_bet]
        coef = (self.means_ - self.xbar_) @ self.scalings_
        self.intercept_ = -0.5 * np.sum(coef**2, axis=1) + np.log(self.priors_)
        self.coef_ = coef @ self.scalings_.T
        self.intercept_ -= self.xbar_ @ self.coef_.T

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_min_samples=2)
        self.classes_ = unique_labels(y)
        n_samples, n_features = X.shape
        n_classes = self.classes_.shape[0]
        if self.priors is None:
            _, cnts = np.unique(y, return_counts=True)
            self.priors_ = cnts.astype(float) / n_samples
        else:
            self.priors_ = np.asarray(self.priors, dtype=float)
        max_components = min(n_classes - 1, n_features)
        self._max_components = self.n_components if self.n_components is not None else max_components
        if self.solver == "svd":
            self._solve_svd(X, y)
        elif self.solver == "lsqr":
            self._solve_lstsq(X, y, shrinkage=self.shrinkage, covariance_estimator=self.covariance_estimator)
        elif self.solver == "eigen":
            self._solve_eigen(X, y, shrinkage=self.shrinkage, covariance_estimator=self.covariance_estimator)
        if len(self.classes_) == 2:
            self.coef_ = (self.coef_[1, :] - self.coef_[0, :]).reshape(1, -1)
            self.intercept_ = (self.intercept_[1] - self.intercept_[0]).reshape(1)
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        decision = self.decision_function(X)
        if len(self.classes_) == 2:
            from scipy.special import expit
            proba = expit(decision)
            return np.stack([1 - proba, proba], axis=1)
        else:
            return softmax(decision)

    def decision_function(self, X):
        check_is_fitted(self)
        X = check_array(X)
        scores = np.dot(X, self.coef_.T) + self.intercept_
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            indices = (scores > 0).astype(int)
        else:
            indices = np.argmax(scores, axis=1)
        return self.classes_[indices]
