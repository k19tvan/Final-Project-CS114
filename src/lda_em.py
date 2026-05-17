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

class LDA_EM(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.5, max_iter=50, tol=1e-4, shrinkage='auto'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.shrinkage = shrinkage

    def fit(self, X_l, y_l, X_u):
        X_l = check_array(X_l)
        X_u = check_array(X_u)
        self.classes_ = np.unique(y_l)
        n_features = X_l.shape[1]
        n_labeled = X_l.shape[0]
        n_unlabeled = X_u.shape[0]
        
        # Initialization: Fit on labeled data
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=self.shrinkage)
        lda.fit(X_l, y_l)
        
        self.priors_ = lda.priors_
        self.means_ = lda.means_
        self.covariance_ = lda.covariance_
        
        prev_log_likelihood = -np.inf
        self.convergence_log_likelihood = []
        
        inv_cov = np.linalg.pinv(self.covariance_)
        
        for i in range(self.max_iter):
            # E-step: Compute responsibilities for unlabeled data
            probs_u = []
            for k, c in enumerate(self.classes_):
                diff = X_u - self.means_[k]
                quad = np.einsum('ij,ij->i', diff @ inv_cov, diff)
                score = np.log(self.priors_[k] + 1e-10) - 0.5 * quad
                probs_u.append(score)
            
            probs_u = np.array(probs_u).T
            probs_u_max = np.max(probs_u, axis=1, keepdims=True)
            exp_probs_u = np.exp(probs_u - probs_u_max)
            responsibilities = exp_probs_u / (np.sum(exp_probs_u, axis=1, keepdims=True) + 1e-10)
            
            # M-step: Update parameters
            total_weight = n_labeled + self.alpha * n_unlabeled
            
            new_priors = np.zeros(len(self.classes_))
            new_means = np.zeros((len(self.classes_), n_features))
            for k, c in enumerate(self.classes_):
                mask_l = (y_l == c)
                w_l = np.sum(mask_l)
                w_u = self.alpha * np.sum(responsibilities[:, k])
                new_priors[k] = (w_l + w_u) / total_weight
                
                sum_l = np.sum(X_l[mask_l], axis=0)
                sum_u = np.sum(responsibilities[:, k][:, np.newaxis] * X_u, axis=0)
                new_means[k] = (sum_l + self.alpha * sum_u) / (w_l + w_u + 1e-10)
            
            cov_acc = np.zeros((n_features, n_features))
            for k, c in enumerate(self.classes_):
                mask_l = (y_l == c)
                diff_l = X_l[mask_l] - new_means[k]
                cov_acc += diff_l.T @ diff_l
                
                diff_u = X_u - new_means[k]
                weighted_diff_u = (responsibilities[:, k] * self.alpha)[:, np.newaxis] * diff_u
                cov_acc += weighted_diff_u.T @ diff_u
            
            new_cov = cov_acc / (total_weight - len(self.classes_))
            
            if self.shrinkage == 'auto':
                new_cov += np.eye(n_features) * 1e-4
            elif isinstance(self.shrinkage, Real):
                new_cov = (1 - self.shrinkage) * new_cov + self.shrinkage * np.mean(np.diag(new_cov)) * np.eye(n_features)

            self.priors_ = new_priors
            self.means_ = new_means
            self.covariance_ = new_cov
            inv_cov = np.linalg.pinv(self.covariance_)
            
            ll_l = 0
            for k, c in enumerate(self.classes_):
                mask_l = (y_l == c)
                diff_l = X_l[mask_l] - self.means_[k]
                quad_l = np.einsum('ij,ij->i', diff_l @ inv_cov, diff_l)
                ll_l += np.sum(np.log(self.priors_[k] + 1e-10) - 0.5 * quad_l)
            
            ll_u = np.sum(np.log(np.sum(np.exp(probs_u - probs_u_max), axis=1) + 1e-10) + probs_u_max.flatten())
            current_log_likelihood = ll_l + self.alpha * ll_u
            self.convergence_log_likelihood.append(current_log_likelihood)
            
            if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = current_log_likelihood
            
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        inv_cov = np.linalg.pinv(self.covariance_)
        probs = []
        for k, c in enumerate(self.classes_):
            diff = X - self.means_[k]
            quad = np.einsum('ij,ij->i', diff @ inv_cov, diff)
            score = np.log(self.priors_[k] + 1e-10) - 0.5 * quad
            probs.append(score)
        
        probs = np.array(probs).T
        probs_max = np.max(probs, axis=1, keepdims=True)
        exp_probs = np.exp(probs - probs_max)
        return exp_probs / (np.sum(exp_probs, axis=1, keepdims=True) + 1e-10)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

class GLA(BaseEstimator, ClassifierMixin):
    def __init__(self, shrinkage='auto'):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_samples = X.shape[0]

        self.priors_ = np.array([np.mean(y == c) for c in self.classes_])
        self.means_ = np.array([np.mean(X[y == c], axis=0) for c in self.classes_])

        # Pooled covariance
        cov_acc = np.zeros((n_features, n_features))
        for k, c in enumerate(self.classes_):
            diff = X[y == c] - self.means_[k]
            cov_acc += diff.T @ diff
        
        self.covariance_ = cov_acc / (n_samples - len(self.classes_))

        if self.shrinkage == 'auto':
            self.covariance_ += np.eye(n_features) * 1e-4
        elif isinstance(self.shrinkage, Real):
            self.covariance_ = (1 - self.shrinkage) * self.covariance_ + self.shrinkage * np.mean(np.diag(self.covariance_)) * np.eye(n_features)

        self.inv_cov_ = np.linalg.pinv(self.covariance_)
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        probs = []
        for k, c in enumerate(self.classes_):
            diff = X - self.means_[k]
            quad = np.einsum('ij,ij->i', diff @ self.inv_cov_, diff)
            score = np.log(self.priors_[k] + 1e-10) - 0.5 * quad
            probs.append(score)
        
        probs = np.array(probs).T
        probs_max = np.max(probs, axis=1, keepdims=True)
        exp_probs = np.exp(probs - probs_max)
        return exp_probs / (np.sum(exp_probs, axis=1, keepdims=True) + 1e-10)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
