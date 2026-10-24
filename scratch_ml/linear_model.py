"""Linear and generalized linear models implemented from scratch.

Math summary
------------
**Linear regression** models ``y = X w + b + eps`` and minimises the
mean squared error

    ``J(w, b) = (1/n) * sum_i (x_i^T w + b - y_i)^2``

whose gradient is ``grad_w = (2/n) X^T (Xw + b - y)`` and
``grad_b = (2/n) * sum_i (x_i^T w + b - y_i)``.  Setting the gradient
to zero gives the *normal equation*
``theta = (X^T X)^{-1} X^T y`` (with a column of ones for the
intercept), which this module also supports via a stable least-squares
solve.

**Logistic regression** models ``P(y = 1 | x) = sigma(z)`` with
``z = x^T w + b`` and the sigmoid ``sigma(z) = 1 / (1 + e^{-z})``.
Training minimises the binary cross-entropy (plus an L2 penalty on the
weights, never on the intercept)

    ``J = -(1/n) sum_i [y_i log p_i + (1 - y_i) log(1 - p_i)]
          + (lambda / 2) * ||w||^2``

with gradients

    ``grad_w = (1/n) X^T (p - y) + lambda * w``,
    ``grad_b = (1/n) sum_i (p_i - y_i)``.
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator
from .datasets import make_classification, make_regression
from .metrics import accuracy_score, mean_squared_error, r2_score
from .utils import (
    LabelMapper,
    check_X,
    check_X_y,
    sigmoid,
    train_test_split,
)

__all__ = ["ScratchLinearRegression", "ScratchLogisticRegression"]


class ScratchLinearRegression(BaseEstimator):
    """Ordinary least squares via gradient descent or the normal equation.

    Parameters
    ----------
    learning_rate : float
        Step size used by batch gradient descent (ignored when
        ``method="normal"``).
    n_iterations : int
        Number of full-batch gradient descent iterations.
    method : {"gradient_descent", "normal"}
        ``"gradient_descent"`` minimises MSE with the update
        ``w <- w - lr * grad_w``; ``"normal"`` solves the closed-form
        system ``theta = (X^T X)^-1 X^T y`` using
        ``numpy.linalg.lstsq`` (an SVD-based solve that is more
        numerically stable than explicitly inverting ``X^T X``).
    fit_intercept : bool
        Whether to estimate an intercept term ``b``.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Learned weights ``w``.
    intercept_ : float
        Learned intercept ``b`` (0.0 when ``fit_intercept=False``).
    loss_history_ : ndarray of shape (n_iterations,)
        Training MSE recorded at every gradient descent iteration (a
        single-element array holding the final MSE for ``method="normal"``).
    n_iter_ : int
        Number of iterations actually performed.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        method: str = "gradient_descent",
        fit_intercept: bool = True,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.fit_intercept = fit_intercept

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "ScratchLinearRegression":
        """Fit ``w`` and ``b`` by gradient descent or the normal equation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Continuous targets.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        y = y.astype(float)
        if self.method not in ("gradient_descent", "normal"):
            raise ValueError(
                f"method must be 'gradient_descent' or 'normal', "
                f"got {self.method!r}."
            )

        if self.method == "normal":
            self._fit_normal_equation(X, y)
        else:
            self._fit_gradient_descent(X, y)
        self._mark_fitted()
        return self

    # ------------------------------------------------------------------
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """Solve ``theta = (X^T X)^-1 X^T y`` in one shot.

        An explicit inverse is avoided: ``np.linalg.lstsq`` solves the
        same least-squares system through SVD and degrades gracefully
        for collinear features.
        """
        design = self._design_matrix(X)
        theta, *_ = np.linalg.lstsq(design, y, rcond=None)
        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta
        self.loss_history_ = np.array(
            [float(np.mean((design @ theta - y) ** 2))]
        )
        self.n_iter_ = 0

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Batch gradient descent on the MSE surface."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        intercept = 0.0
        losses = []

        for iteration in range(1, self.n_iterations + 1):
            residual = X @ weights + intercept - y  # y_hat - y
            losses.append(float(np.mean(residual**2)))
            if not np.isfinite(losses[-1]):
                raise RuntimeError(
                    f"Loss became non-finite at iteration {iteration}; "
                    f"lower learning_rate (currently {self.learning_rate})."
                )
            grad_w = (2.0 / n_samples) * (X.T @ residual)
            weights -= self.learning_rate * grad_w
            if self.fit_intercept:
                grad_b = (2.0 / n_samples) * float(np.sum(residual))
                intercept -= self.learning_rate * grad_b

        self.coef_ = weights
        self.intercept_ = float(intercept) if self.fit_intercept else 0.0
        self.loss_history_ = np.asarray(losses, dtype=float)
        self.n_iter_ = iteration

    # ------------------------------------------------------------------
    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Prepend a column of ones when an intercept is requested."""
        if not self.fit_intercept:
            return X
        return np.column_stack([np.ones(X.shape[0]), X])

    def predict(self, X) -> np.ndarray:
        """Return predictions ``X @ coef_ + intercept_``."""
        self._check_is_fitted()
        X = check_X(X)
        return X @ self.coef_ + self.intercept_


class ScratchLogisticRegression(BaseEstimator):
    """Binary logistic regression trained with gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for batch gradient descent.
    n_iterations : int
        Number of training epochs.
    l2 : float
        Ridge (L2) penalty strength ``lambda`` applied to the weights
        only.  ``0.0`` disables regularization.
    fit_intercept : bool
        Whether to estimate an intercept ``b`` (never penalized).
    threshold : float
        Decision threshold on ``P(y = 1 | x)`` used by :meth:`predict`.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Learned weights ``w``.
    intercept_ : float
        Learned intercept ``b``.
    classes_ : ndarray of shape (2,)
        The two class labels seen during :meth:`fit` (sorted).
    loss_history_ : ndarray of shape (n_iterations,)
        Penalized binary cross-entropy recorded every epoch.
    n_iter_ : int
        Number of epochs performed.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 1000,
        l2: float = 0.0,
        fit_intercept: bool = True,
        threshold: float = 0.5,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.threshold = threshold

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "ScratchLogisticRegression":
        """Fit the model by minimising penalized binary cross-entropy.

        Targets must contain exactly two distinct labels; they are
        encoded internally so that the positive class is the larger
        (sorted) label.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        if self.l2 < 0:
            raise ValueError("l2 must be >= 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self._mapper = LabelMapper().fit(y)
        self.classes_ = self._mapper.classes_
        if self.classes_.size != 2:
            raise ValueError(
                f"ScratchLogisticRegression is binary; got "
                f"{self.classes_.size} classes: {self.classes_}."
            )
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be >= 1.")

        target = self._mapper.transform(y).astype(float)  # {0, 1}
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        intercept = 0.0
        losses = []

        for iteration in range(1, self.n_iterations + 1):
            probabilities = sigmoid(X @ weights + intercept)
            # Binary cross-entropy with clipped probabilities.
            eps = 1e-15
            p = np.clip(probabilities, eps, 1.0 - eps)
            bce = -float(
                np.mean(target * np.log(p)
                        + (1.0 - target) * np.log(1.0 - p))
            )
            penalty = 0.5 * self.l2 * float(weights @ weights)
            losses.append(bce + penalty)

            error = probabilities - target
            grad_w = (X.T @ error) / n_samples + self.l2 * weights
            grad_b = float(np.mean(error)) if self.fit_intercept else 0.0
            weights -= self.learning_rate * grad_w
            intercept -= self.learning_rate * grad_b

        self.coef_ = weights
        self.intercept_ = float(intercept) if self.fit_intercept else 0.0
        self.loss_history_ = np.asarray(losses, dtype=float)
        self.n_iter_ = iteration
        self._mark_fitted()
        return self

    # ------------------------------------------------------------------
    def decision_function(self, X) -> np.ndarray:
        """Raw linear scores ``z = X @ coef_ + intercept_``."""
        self._check_is_fitted()
        return check_X(X) @ self.coef_ + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities of shape ``(n_samples, 2)``.

        Column order follows :attr:`classes_`, i.e. column 1 holds
        ``P(y = classes_[1] | x)``.
        """
        p_positive = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p_positive, p_positive])

    def predict(self, X) -> np.ndarray:
        """Return the label with probability above :attr:`threshold`."""
        probabilities = self.predict_proba(X)[:, 1]
        positive = (probabilities >= self.threshold).astype(int)
        return self._mapper.inverse_transform(positive)


def demo() -> None:
    """Synthetic end-to-end checks for both linear models."""
    print("=" * 62)
    print("ScratchLinearRegression / ScratchLogisticRegression")
    print("=" * 62)

    # --- Linear regression: compare GD against the normal equation ----
    X, y = make_regression(
        n_samples=400, n_features=4, noise=0.3, bias=1.5, random_state=0
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    gd = ScratchLinearRegression(
        learning_rate=0.05, n_iterations=2000, method="gradient_descent"
    ).fit(X_tr, y_tr)
    ne = ScratchLinearRegression(method="normal").fit(X_tr, y_tr)

    print("[LinearRegression] method=gradient_descent")
    print(f"  train MSE: {mean_squared_error(y_tr, gd.predict(X_tr)):.5f} | "
          f"test MSE: {mean_squared_error(y_te, gd.predict(X_te)):.5f}")
    print(f"  test R^2 : {r2_score(y_te, gd.predict(X_te)):.5f} | "
          f"loss first->last: {gd.loss_history_[0]:.4f} -> "
          f"{gd.loss_history_[-1]:.4f}")
    print("[LinearRegression] method=normal equation")
    print(f"  train MSE: {mean_squared_error(y_tr, ne.predict(X_tr)):.5f} | "
          f"test MSE: {mean_squared_error(y_te, ne.predict(X_te)):.5f}")
    print(f"  recovered weights: {np.round(ne.coef_, 3)} "
          f"(intercept {ne.intercept_:.3f})")

    # --- Logistic regression with an L2 penalty ------------------------
    X, y = make_classification(
        n_samples=400, n_features=2, n_classes=2, class_sep=2.0,
        class_std=1.0, random_state=1,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    clf = ScratchLogisticRegression(
        learning_rate=0.5, n_iterations=1500, l2=0.01
    ).fit(X_tr, y_tr)
    print("[LogisticRegression] sigmoid + BCE, L2=0.01")
    print(f"  train accuracy: {accuracy_score(y_tr, clf.predict(X_tr)):.4f} | "
          f"test accuracy: {accuracy_score(y_te, clf.predict(X_te)):.4f}")
    print(f"  BCE first -> last: {clf.loss_history_[0]:.4f} -> "
          f"{clf.loss_history_[-1]:.4f}")
    print(f"  weights: {np.round(clf.coef_, 3)}, intercept: "
          f"{clf.intercept_:.3f}")


if __name__ == "__main__":
    demo()
