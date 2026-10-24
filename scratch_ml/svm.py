"""Linear Support Vector Machine implemented from scratch.

Math summary
------------
Labels are recoded to ``y_i in {-1, +1}`` and the linear score is
``f(x) = x^T w + b``.  Training minimizes the soft-margin objective

    ``J(w, b) = 1/2 ||w||^2 + (C / n) * sum_i max(0, 1 - y_i f(x_i))``

* ``1/2 ||w||^2`` is the *regularization* term: shrinking ``||w||``
  maximizes the margin ``2 / ||w||`` between the two class
  hyperplanes ``x^T w + b = +/-1``.
* ``max(0, 1 - y_i f(x_i))`` is the *hinge loss*: it penalizes points
  inside the margin (``y_i f_i < 1``) or on the wrong side.
* ``C`` trades margin size off against training errors (larger ``C`` =
  harder margin).

The subgradient of the hinge loss is

    ``dJ/dw = w - (C / n) * sum_{i : y_i f_i < 1} y_i x_i``
    ``dJ/db = -(C / n) * sum_{i : y_i f_i < 1} y_i``

and both are applied with batch gradient (subgradient) descent.
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator
from .datasets import make_classification
from .metrics import accuracy_score
from .utils import LabelMapper, check_X, check_X_y, train_test_split

__all__ = ["ScratchLinearSVM"]


class ScratchLinearSVM(BaseEstimator):
    """Linear SVM trained with hinge loss by batch subgradient descent.

    Parameters
    ----------
    C : float
        Penalty weight for margin violations (soft-margin strength).
        Must be > 0; larger values punish violations harder.
    learning_rate : float
        Step size of the subgradient descent updates.
    n_iterations : int
        Number of full-batch epochs.
    fit_intercept : bool
        Whether to learn a bias ``b`` (never regularized).

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Weight vector ``w`` of the separating hyperplane.
    intercept_ : float
        Bias ``b``.
    classes_ : ndarray of shape (2,)
        Original class labels (sorted); predictions are mapped back
        from the internal ``{-1, +1}`` encoding.
    support_mask_ : ndarray of shape (n_samples,)
        Boolean mask of training points that lie inside the margin or
        are misclassified (``y_i * f_i < 1``) after training.
    loss_history_ : ndarray of shape (n_iterations,)
        Objective value ``J(w, b)`` recorded every epoch.
    """

    def __init__(
        self,
        C: float = 1.0,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
    ):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "ScratchLinearSVM":
        """Fit the soft-margin hyperplane with subgradient descent.

        Targets must be binary; they are recoded to ``{-1, +1}``.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        if self.C <= 0:
            raise ValueError("C must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be >= 1.")

        self._mapper = LabelMapper().fit(y)
        self.classes_ = self._mapper.classes_
        if self.classes_.size != 2:
            raise ValueError(
                f"ScratchLinearSVM is binary; got {self.classes_.size} "
                f"classes: {self.classes_}."
            )
        # {0, 1} -> {-1, +1}
        encoded = 2.0 * self._mapper.transform(y).astype(float) - 1.0

        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        intercept = 0.0
        losses = []

        for _ in range(1, self.n_iterations + 1):
            margins = encoded * (X @ weights + intercept)  # y_i * f_i
            violated = margins < 1.0

            # Objective value: reg + mean hinge penalty.
            hinge = np.maximum(0.0, 1.0 - margins)
            losses.append(
                0.5 * float(weights @ weights)
                + (self.C / n_samples) * float(np.sum(hinge))
            )
            if not np.isfinite(losses[-1]):
                raise RuntimeError(
                    "Loss became non-finite; lower learning_rate "
                    f"(currently {self.learning_rate})."
                )

            # Subgradient of (C / n) * sum hinge  +  1/2 ||w||^2.
            count = int(np.sum(violated))
            grad_w = weights.copy()
            grad_b = 0.0
            if count:
                grad_w -= (self.C / n_samples) * (
                    X[violated].T @ encoded[violated]
                )
                if self.fit_intercept:
                    grad_b = -(self.C / n_samples) * float(
                        np.sum(encoded[violated])
                    )
            weights -= self.learning_rate * grad_w
            if self.fit_intercept:
                intercept -= self.learning_rate * grad_b

        self.coef_ = weights
        self.intercept_ = float(intercept) if self.fit_intercept else 0.0
        self.loss_history_ = np.asarray(losses, dtype=float)
        self.n_iter_ = self.n_iterations
        margins = encoded * (X @ self.coef_ + self.intercept_)
        self.support_mask_ = margins < 1.0
        self._mark_fitted()
        return self

    # ------------------------------------------------------------------
    def decision_function(self, X) -> np.ndarray:
        """Signed distance score ``f(x) = X @ coef_ + intercept_``."""
        self._check_is_fitted()
        return check_X(X) @ self.coef_ + self.intercept_

    def predict(self, X) -> np.ndarray:
        """Predict labels by the sign of the decision function."""
        self._check_is_fitted()
        positive = (self.decision_function(X) >= 0.0).astype(int)
        return self._mapper.inverse_transform(positive)


def demo() -> None:
    """Synthetic end-to-end check on separable Gaussian blobs."""
    print("=" * 62)
    print("ScratchLinearSVM (hinge loss + gradient descent)")
    print("=" * 62)

    X, y = make_classification(
        n_samples=400, n_features=2, n_classes=2, class_sep=2.5,
        class_std=1.0, random_state=5,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=5)

    svm = ScratchLinearSVM(
        C=1.0, learning_rate=0.01, n_iterations=1500
    ).fit(X_tr, y_tr)

    print(f"  train accuracy: {accuracy_score(y_tr, svm.predict(X_tr)):.4f}"
          f" | test accuracy: {accuracy_score(y_te, svm.predict(X_te)):.4f}")
    print(f"  objective first -> last: {svm.loss_history_[0]:.4f} -> "
          f"{svm.loss_history_[-1]:.4f}")
    print(f"  margin violations on train: {svm.support_mask_.sum()} / "
          f"{X_tr.shape[0]}")
    print(f"  ||w|| = {np.linalg.norm(svm.coef_):.4f} -> geometric margin "
          f"~ {2.0 / np.linalg.norm(svm.coef_):.4f}")


if __name__ == "__main__":
    demo()
