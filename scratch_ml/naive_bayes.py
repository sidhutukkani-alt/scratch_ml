"""Gaussian Naive Bayes implemented from scratch.

Math summary
------------
Bayes' rule with the *naive* conditional-independence assumption:

    ``P(y = c | x)  ~  P(y = c) * prod_j P(x_j | y = c)``

Each class-conditional density is modeled as an independent Gaussian
per feature,

    ``P(x_j | y = c) = N(x_j; mu_cj, sigma^2_cj)
                     = 1 / sqrt(2 pi sigma^2_cj)
                       * exp(-(x_j - mu_cj)^2 / (2 sigma^2_cj))``

so :meth:`fit` only needs per-class means/variances plus the class
priors.  Prediction is done in log space to avoid underflow:

    ``log P(c | x) = log P(c) - 0.5 sum_j [log(2 pi sigma^2_cj)
                       + (x_j - mu_cj)^2 / sigma^2_cj] + const``

and the argmax over ``c`` is the predicted class.  A tiny variance
floor ("variance smoothing") keeps zero-variance features from
producing division by zero.
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator
from .datasets import make_blobs
from .metrics import accuracy_score
from .utils import check_X, check_X_y, train_test_split

__all__ = ["ScratchNaiveBayes"]


class ScratchNaiveBayes(BaseEstimator):
    """Gaussian Naive Bayes classifier.

    Parameters
    ----------
    var_smoothing : float
        Variance floor expressed as a fraction of the largest feature
        variance in ``X``: every class variance is clamped to at least
        ``max(var_smoothing * max_var, eps)``.  Defaults to ``1e-10``,
        matching scikit-learn's ``GaussianNB``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted unique class labels.
    priors_ : ndarray of shape (n_classes,)
        Empirical class probabilities ``P(y = c)``.
    means_ : ndarray of shape (n_classes, n_features)
        Per-class feature means ``mu_cj``.
    variances_ : ndarray of shape (n_classes, n_features)
        Per-class feature variances ``sigma^2_cj`` (after smoothing).
    var_smoothing_ : float
        The actual epsilon applied during :meth:`fit`.
    """

    def __init__(self, var_smoothing: float = 1e-10):
        self.var_smoothing = var_smoothing

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "ScratchNaiveBayes":
        """Estimate priors, means and variances for every class.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        if self.var_smoothing < 0:
            raise ValueError("var_smoothing must be >= 0.")

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = self.classes_.size

        counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)
        if np.any(counts == 0):
            raise ValueError("Encountered an empty class during fit.")
        self.priors_ = counts / n_samples

        self.means_ = np.empty((n_classes, n_features))
        self.variances_ = np.empty((n_classes, n_features))
        for idx, label in enumerate(self.classes_):
            class_X = X[y == label]
            self.means_[idx] = class_X.mean(axis=0)
            self.variances_[idx] = class_X.var(axis=0, ddof=0)

        # Variance smoothing: floor tiny variances to avoid 1/0.
        max_var = float(np.var(X, axis=0).max()) if n_samples > 1 else 0.0
        self.var_smoothing_ = max(
            self.var_smoothing * max_var, np.finfo(float).eps
        )
        self.variances_ = np.maximum(self.variances_, self.var_smoothing_)

        self._mark_fitted()
        return self

    # ------------------------------------------------------------------
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """``log P(c) + log P(x | c)`` for every class ``c``.

        Shape: ``(n_samples, n_classes)``.
        """
        log_prior = np.log(self.priors_)
        joint = np.empty((X.shape[0], self.classes_.size))
        for idx in range(self.classes_.size):
            mean = self.means_[idx]
            var = self.variances_[idx]
            # log N(x_j; mu, sigma^2) = -0.5*(log(2 pi sigma^2)
            #                            + (x_j - mu)^2 / sigma^2)
            log_det = np.sum(np.log(2.0 * np.pi * var))
            maha = np.sum((X - mean) ** 2 / var, axis=1)
            joint[:, idx] = log_prior[idx] - 0.5 * (log_det + maha)
        return joint

    def predict_log_proba(self, X) -> np.ndarray:
        """Log of class probabilities via log-sum-exp normalization."""
        self._check_is_fitted()
        joint = self._joint_log_likelihood(check_X(X))
        max_log = joint.max(axis=1, keepdims=True)
        shifted = np.exp(joint - max_log)
        normalizer = np.log(shifted.sum(axis=1, keepdims=True)) + max_log
        return joint - normalizer

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities of shape ``(n_samples, n_classes)``."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        """Return the class with the highest posterior probability."""
        self._check_is_fitted()
        joint = self._joint_log_likelihood(check_X(X))
        return self.classes_[np.argmax(joint, axis=1)]


def demo() -> None:
    """Synthetic end-to-end check on Gaussian blobs."""
    print("=" * 62)
    print("ScratchNaiveBayes (Gaussian)")
    print("=" * 62)

    centers = np.array([[0.0, 0.0], [3.5, 0.5], [0.5, 3.5]])
    X, y = make_blobs(n_samples=450, centers=centers, n_features=2,
                      cluster_std=1.3, random_state=4)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=4)
    clf = ScratchNaiveBayes().fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)

    print(f"  classes: {clf.classes_} | priors: "
          f"{np.round(clf.priors_, 3)}")
    print(f"  train accuracy: {accuracy_score(y_tr, clf.predict(X_tr)):.4f}"
          f" | test accuracy: {accuracy_score(y_te, clf.predict(X_te)):.4f}")
    print(f"  mean predicted probability: {proba.mean(axis=0).round(3)}")


if __name__ == "__main__":
    demo()
