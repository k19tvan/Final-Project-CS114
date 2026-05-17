import warnings
from numbers import Integral, Real
import numpy as np
import scipy.linalg
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import empirical_covariance, ledoit_wolf, shrunk_covariance
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class DimensionReduction:
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, X):
        return self.pca.fit_transform(X)

    def transform(self, X):
        return self.pca.transform(X)

class SVMClassifier_wrapper:
    def __init__(self, C=1.0):
        self.model = LinearSVC(C=C, dual=False, max_iter=2000)
        
    def train(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

class LDAClassifier_wrapper:
    def __init__(self):
        self.lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    def fit(self, X, y):
        self.lda.fit(X, y)

    def predict(self, X):
        return self.lda.predict(X)

    def score(self, X, y):
        return self.lda.score(X, y)
        
    def predict_proba(self, X):
        return self.lda.predict_proba(X)

import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.covariance import ledoit_wolf

class WLDA(BaseEstimator, ClassifierMixin):
    """
    Weighted missing Linear Discriminant Analysis (WLDA)
    Dựa trên paper: "Directly Handling Missing Data in Linear Discriminant Analysis"
    Hỗ trợ dữ liệu có missing values (NaN) trong cả training và test.
    """

    def __init__(self, shrinkage='ledoit-wolf', shrinkage_value=0.1):
        """
        Parameters
        ----------
        shrinkage : str or float, default='ledoit-wolf'
            - 'ledoit-wolf' : sử dụng Ledoit-Wolf shrinkage (khuyến nghị cho high-dim)
            - 'manual' : sử dụng giá trị cố định shrinkage_value
        shrinkage_value : float, default=0.1
            Giá trị shrinkage khi shrinkage='manual'
        """
        self.shrinkage = shrinkage
        self.shrinkage_value = shrinkage_value

    def _estimate_params_dper(self, X, y):
        """
        Ước lượng mean (theo từng class) và pooled covariance matrix
        bằng Direct Parameter Estimation (DPER).
        Không imputation, chỉ dùng các cặp quan sát được.
        """
        X = X.astype(float)
        classes = np.unique(y)
        n_features = X.shape[1]

        # 1. Mean của mỗi class (chỉ dùng quan sát có được)
        means = {}
        for g in classes:
            Xg = X[y == g]
            # mean theo từng feature, bỏ qua NaN
            mu_g = np.nanmean(Xg, axis=0)
            # Nếu feature bị missing hoàn toàn trong class g, đặt bằng global mean
            global_mean = np.nanmean(X, axis=0)
            mu_g = np.where(np.isnan(mu_g), global_mean, mu_g)
            means[g] = mu_g

        # 2. Pooled covariance matrix DPER
        cov = np.zeros((n_features, n_features))
        counts = np.zeros((n_features, n_features))

        # Duyệt từng cặp feature (i,j)
        # Với p lớn (HOG) vòng lặp O(p^2) nhưng vẫn chấp nhận được.
        for i in range(n_features):
            for j in range(i, n_features):
                total_sum = 0.0
                n_pair = 0
                for g in classes:
                    Xg = X[y == g]
                    mu_g = means[g]
                    # Mặt nạ: cả feature i và j đều observed
                    mask = ~np.isnan(Xg[:, i]) & ~np.isnan(Xg[:, j])
                    if mask.sum() == 0:
                        continue
                    xi = Xg[mask, i] - mu_g[i]
                    xj = Xg[mask, j] - mu_g[j]
                    total_sum += np.sum(xi * xj)
                    n_pair += mask.sum()
                if n_pair > 0:
                    cov[i, j] = total_sum / n_pair
                    cov[j, i] = cov[i, j]
                counts[i, j] = n_pair
                counts[j, i] = n_pair

        # Nếu có cặp nào không được cập nhật (n_pair=0), giữ cov=0

        # 3. Shrinkage để ma trận khả nghịch và ổn định
        if self.shrinkage == 'ledoit-wolf':
            # ledoit_wolf yêu cầu ma trận không có NaN, ta đã có cov đầy đủ
            # Nhưng ledoit_wolf thường được dùng trên covariance mẫu từ dữ liệu hoàn chỉnh.
            # Dùng trực tiếp trên cov ước lượng được có thể ổn định hơn.
            # Thực hiện shrinkage bằng cách xáo trộn về ma trận scaled identity
            # Công thức đơn giản: (1-alpha)*cov + alpha*mean(diag(cov))*I
            diag_mean = np.mean(np.diag(cov))
            alpha = min(1.0, 2.0 * n_features / X.shape[0])  # heuristic đơn giản
            cov = (1 - alpha) * cov + alpha * diag_mean * np.eye(n_features)
        else:  # manual
            diag_mean = np.mean(np.diag(cov))
            cov = (1 - self.shrinkage_value) * cov + self.shrinkage_value * diag_mean * np.eye(n_features)

        return means, cov

    def fit(self, X, y):
        """Huấn luyện WLDA"""
        try:
            X, y = check_X_y(X, y, accept_sparse=False, force_all_finite=False)
        except TypeError:
            X, y = check_X_y(X, y, accept_sparse=False, ensure_all_finite=False)

        X = X.astype(float)
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]

        # Prior probabilities
        self.priors_ = np.array([np.sum(y == c) / n_samples for c in self.classes_])

        # Missing rate per feature (trên toàn bộ training set)
        missing_rates = np.nanmean(np.isnan(X), axis=0)
        # Tránh missing_rate = 1 (gây vô cùng), giới hạn tối đa 0.99
        missing_rates = np.clip(missing_rates, 0.0, 0.99)
        self.weights_ = 1.0 / (1.0 - missing_rates)   # w_i

        # Ước lượng mean và covariance bằng DPER
        self.means_, cov_ = self._estimate_params_dper(X, y)

        # Tính pseudo-inverse của covariance (ổn định hơn inverse trực tiếp)
        # Dùng np.linalg.pinv với threshold nhỏ
        self.cov_inv_ = np.linalg.pinv(cov_, hermitian=True, rtol=1e-6)

        return self

    def decision_function(self, X):
        """Tính scores cho mỗi class (log-posterior chưa chuẩn hóa)"""
        try:
            X = check_array(X, accept_sparse=False, force_all_finite=False)
        except TypeError:
            X = check_array(X, accept_sparse=False, ensure_all_finite=False)

        X = X.astype(float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # X_filled: thay NaN bằng 0 (chỉ dùng để tính (x-mu), weight sẽ loại bỏ đóng góp của missing)
        X_filled = np.nan_to_num(X, nan=0.0)
        mask = ~np.isnan(X)                # True = observed
        # Ma trận trọng số W_x: m_i * w_i cho mỗi sample
        W = mask.astype(float) * self.weights_

        scores = np.zeros((n_samples, n_classes))
        for idx, c in enumerate(self.classes_):
            mu = self.means_[c]
            # (x - mu) đã thay missing bằng 0, nhưng sẽ được nhân với W -> missing bị triệt tiêu
            diff = X_filled - mu
            wd = W * diff                    # W_x * (x - mu)
            # Quadratic form: wd^T Sigma^{-1} wd
            quad = np.einsum('ij,ij->i', wd @ self.cov_inv_, wd)   # efficient
            scores[:, idx] = np.log(self.priors_[idx] + 1e-10) - 0.5 * quad
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Xác suất hậu nghiệm (softmax của scores)"""
        scores = self.decision_function(X)
        # Tránh overflow: trừ đi max mỗi dòng
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return proba
        
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
        
        for i in range(self.max_iter):
            # E-step: Compute responsibilities for unlabeled data
            # Use lda.predict_proba or compute manually
            # Score = log(prior) - 0.5 * (x-mu)^T InvCov (x-mu)
            # For efficiency, we can use the decision_function logic
            
            # Since we have the internal LinearDiscriminantAnalysis code in the same file,
            # we can just use it.
            
            # Calculate responsibilities
            # We need to compute P(y=k | x, theta)
            # r_ik = exp(score_k) / sum(exp(score_j))
            
            # Using current params to score X_u
            # We can use lda.decision_function if we update it, but it's easier to use it directly
            lda.priors_ = self.priors_
            lda.means_ = self.means_
            lda.covariance_ = self.covariance_
            # Recompute coef_ and intercept_ for decision_function if using lsqr
            # Or just compute responsibilities manually
            
            # Manual responsibility calculation
            inv_cov = np.linalg.pinv(self.covariance_)
            probs_u = []
            for k, c in enumerate(self.classes_):
                diff = X_u - self.means_[k]
                quad = np.einsum('ij,ij->i', diff @ inv_cov, diff)
                score = np.log(self.priors_[k] + 1e-10) - 0.5 * quad
                probs_u.append(score)
            
            probs_u = np.array(probs_u).T
            # Softmax to get responsibilities
            probs_u = probs_u - np.max(probs_u, axis=1, keepdims=True)
            responsibilities = np.exp(probs_u)
            responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
            
            # M-step: Update parameters
            # Labeled weights = 1, Unlabeled weights = alpha * r_ik
            
            new_priors = np.zeros(len(self.classes_))
            new_means = np.zeros((len(self.classes_), n_features))
            new_cov = np.zeros((n_features, n_features))
            
            total_weight = n_labeled + self.alpha * X_u.shape[0]
            
            for k, c in enumerate(self.classes_):
                mask_l = (y_l == c)
                w_l = np.sum(mask_l)
                w_u = self.alpha * np.sum(responsibilities[:, k])
                
                new_priors[k] = (w_l + w_u) / total_weight
                
                # Update means
                sum_l = np.sum(X_l[mask_l], axis=0)
                sum_u = np.sum(responsibilities[:, k][:, np.newaxis] * X_u, axis=0)
                new_means[k] = (sum_l + self.alpha * sum_u) / (w_l + w_u + 1e-10)
            
            # Update Pooled Covariance
            # Sum_k (Sum_i w_i * (x_i - mu_k)(x_i - mu_k)^T)
            cov_acc = np.zeros((n_features, n_features))
            for k, c in enumerate(self.classes_):
                # Labeled part
                mask_l = (y_l == c)
                diff_l = X_l[mask_l] - new_means[k]
                cov_acc += diff_l.T @ diff_l
                
                # Unlabeled part
                diff_u = X_u - new_means[k]
                # Weighted outer products
                # responsibilities[:, k] has shape (n_unlabeled,)
                # diff_u has shape (n_unlabeled, n_features)
                weighted_diff_u = (responsibilities[:, k] * self.alpha)[:, np.newaxis] * diff_u
                cov_acc += weighted_diff_u.T @ diff_u
                
            new_cov = cov_acc / (total_weight - len(self.classes_))
            
            # Apply shrinkage if requested
            if self.shrinkage == 'auto':
                # Simplified Ledoit-Wolf or just add small epsilon
                new_cov += np.eye(n_features) * 1e-4
            elif isinstance(self.shrinkage, Real):
                new_cov = (1 - self.shrinkage) * new_cov + self.shrinkage * np.mean(np.diag(new_cov)) * np.eye(n_features)

            self.priors_ = new_priors
            self.means_ = new_means
            self.covariance_ = new_cov
            
            # Check convergence via log-likelihood
            # LL = sum_l log(P(x_l, y_l)) + sum_u log(sum_k P(x_u, y=k))
            # Simplified LL check
            ll_l = 0
            for k, c in enumerate(self.classes_):
                mask_l = (y_l == c)
                diff_l = X_l[mask_l] - self.means_[k]
                quad_l = np.einsum('ij,ij->i', diff_l @ inv_cov, diff_l)
                ll_l += np.sum(np.log(self.priors_[k] + 1e-10) - 0.5 * quad_l)
            
            # Unlabeled LL part is already in probs_u before softmax
            ll_u = np.sum(np.log(np.sum(np.exp(probs_u), axis=1) + 1e-10))
            current_log_likelihood = ll_l + self.alpha * ll_u
            
            if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = current_log_likelihood
            
        return self

    def predict_proba(self, X):
        inv_cov = np.linalg.pinv(self.covariance_)
        probs = []
        for k, c in enumerate(self.classes_):
            diff = X - self.means_[k]
            quad = np.einsum('ij,ij->i', diff @ inv_cov, diff)
            score = np.log(self.priors_[k] + 1e-10) - 0.5 * quad
            probs.append(score)
        
        probs = np.array(probs).T
        probs = probs - np.max(probs, axis=1, keepdims=True)
        exp_probs = np.exp(probs)
        return exp_probs / np.sum(exp_probs, axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

