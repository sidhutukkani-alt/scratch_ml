"""Ensemble methods: bagging and gradient boosting, from scratch.

Math summary
------------
**Random forest (bagging).**  Build ``B`` trees on bootstrap samples
``S_b`` (``n`` training points drawn *with replacement*) and, at every
split, only consider a random subset of ``max_features`` features.
Class probabilities are averaged across trees

    ``P(y = c | x) = (1 / B) * sum_b P_b(y = c | x)``

and regression predictions are averaged:

    ``f(x) = (1 / B) * sum_b f_b(x)``

Bagging reduces **variance** because the individual trees' errors are
partially uncorrelated; the per-split feature subset decorrelates the
trees further.

**Gradient boosting.**  Fits ``M`` shallow trees *sequentially* to the
negative gradients (pseudo-residuals) of the loss with respect to the
current additive model, then adds them scaled by a learning rate
``nu``::

    F_0(x)  = argmin_c  sum_i L(y_i, c)          (initial constant)
    r_m     = -dL(y_i, F_{m-1}(x_i)) / dF        (pseudo-residuals)
    h_m     = regression_tree(X, r_m)
    F_m(x)  = F_{m-1}(x) + nu * h_m(x)

For squared loss ``L = (y - F)^2 / 2`` the residual is ``r = y - F``;
for binary logistic loss with ``F`` on the log-odds scale,
``p = sigmoid(F)`` and ``r = y - p``.
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator
from .datasets import make_moons, make_regression
from .metrics import accuracy_score, mean_squared_error, r2_score
from .tree import ScratchDecisionTree
from .utils import (
    LabelMapper,
    check_X,
    check_X_y,
    infer_task,
    sigmoid,
    train_test_split,
)

__all__ = ["ScratchRandomForest", "ScratchGradientBoosting"]


class ScratchRandomForest(BaseEstimator):
    """Bagged ensemble of :class:`~scratch_ml.tree.ScratchDecisionTree`.

    Parameters
    ----------
    n_estimators : int
        Number of trees ``B``.
    criterion : {"gini", "entropy", "mse"} or None
        Impurity used by the individual trees (``None`` = task default).
    max_depth : int or None
        Depth limit shared by every tree.
    min_samples_split, min_samples_leaf : int
        Passed through to each tree.
    max_features : {"sqrt", "log2", int, float, None}
        Size of the random feature subset evaluated at each split;
        ``"sqrt"`` is the classic random-forest default.
    bootstrap : bool
        Draw each tree's training set with replacement (bagging).
    task : {"auto", "classification", "regression"}
        ``"auto"`` infers the task from ``y``.
    random_state : int or None
        Master seed; per-tree seeds derive from it so runs reproduce.

    Attributes
    ----------
    estimators_ : list of ScratchDecisionTree
        The fitted trees.
    classes_ : ndarray
        Sorted class labels (classification only).
    """

    def __init__(
        self,
        n_estimators: int = 10,
        criterion=None,
        max_depth=None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features="sqrt",
        bootstrap: bool = True,
        task: str = "auto",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.task = task
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "ScratchRandomForest":
        """Fit ``n_estimators`` trees on bootstrap samples of ``X``.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if self.task not in ("auto", "classification", "regression"):
            raise ValueError(
                "task must be 'auto', 'classification' or 'regression'."
            )

        self._task = infer_task(y) if self.task == "auto" else self.task
        if self._task == "classification":
            self.classes_ = np.unique(y)
        else:
            self.classes_ = None

        self._rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        self.estimators_ = []
        for _ in range(self.n_estimators):
            seed = int(self._rng.integers(0, 2**31 - 1))
            if self.bootstrap:
                # Draw n samples with replacement (bootstrap resample).
                idx = self._rng.integers(0, n_samples, size=n_samples)
            else:
                idx = np.arange(n_samples)
            tree = ScratchDecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                task=self._task,
                random_state=seed,
            ).fit(X[idx], y[idx])
            self.estimators_.append(tree)

        self._mark_fitted()
        return self

    # ------------------------------------------------------------------
    def _aligned_tree_proba(self, tree, X: np.ndarray) -> np.ndarray:
        """Tree probabilities aligned to the forest's class columns.

        A bootstrap sample can miss a rare class, so the tree's own
        ``classes_`` may be a subset of the forest's; the missing
        columns are simply left at zero probability.
        """
        proba = np.zeros((X.shape[0], self.classes_.size))
        positions = np.searchsorted(self.classes_, tree.classes_)
        proba[:, positions] = tree.predict_proba(X)
        return proba

    def predict_proba(self, X) -> np.ndarray:
        """Average of the trees' class probabilities (classification)."""
        self._check_is_fitted()
        if self._task != "classification":
            raise ValueError("predict_proba is only defined for classifiers.")
        X = check_X(X)
        proba = np.mean(
            [self._aligned_tree_proba(tree, X) for tree in self.estimators_],
            axis=0,
        )
        return proba

    def predict(self, X) -> np.ndarray:
        """Majority probability vote (classification) or mean (regression)."""
        self._check_is_fitted()
        X = check_X(X)
        if self._task == "classification":
            votes = self.predict_proba(X)
            return self.classes_[np.argmax(votes, axis=1)]
        predictions = np.mean(
            [tree.predict(X) for tree in self.estimators_], axis=0
        )
        return predictions.astype(float)


class ScratchGradientBoosting(BaseEstimator):
    """Gradient boosting of shallow regression trees.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages ``M``.
    learning_rate : float
        Shrinkage ``nu`` applied to every tree's contribution.
    max_depth : int
        Depth of each base learner; 2-3 keeps them weak.
    task : {"auto", "classification", "regression"}
        ``"auto"`` infers the task from ``y``; classification supports
        exactly two classes (log-loss on the log-odds scale).
    random_state : int or None
        Seed passed to every base tree.

    Attributes
    ----------
    estimators_ : list of ScratchDecisionTree
        Base learners in fitting order.
    init_ : float
        Constant starting prediction ``F_0`` (mean for regression,
        log-odds of the base rate for classification).
    loss_history_ : ndarray of shape (n_estimators,)
        Training MSE (regression) or binary cross-entropy
        (classification) after each stage.
    classes_ : ndarray
        Sorted class labels (classification only).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        task: str = "auto",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.task = task
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "ScratchGradientBoosting":
        """Train the boosting sequence by fitting residual trees.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if self.task not in ("auto", "classification", "regression"):
            raise ValueError(
                "task must be 'auto', 'classification' or 'regression'."
            )

        self._task = infer_task(y) if self.task == "auto" else self.task
        if self._task == "classification":
            return self._fit_classification(X, y)
        return self._fit_regression(X, y)

    # ------------------------------------------------------------------
    def _make_stage_tree(self, seed: int) -> ScratchDecisionTree:
        """A small regression tree used as a boosting base learner."""
        return ScratchDecisionTree(
            criterion="mse",
            max_depth=self.max_depth,
            task="regression",
            random_state=seed,
        )

    def _fit_regression(self, X: np.ndarray, y: np.ndarray):
        y = y.astype(float)
        self.classes_ = None
        self.init_ = float(np.mean(y))
        current = np.full(X.shape[0], self.init_, dtype=float)
        rng = np.random.default_rng(self.random_state)
        losses = []
        self.estimators_ = []

        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, 2**31 - 1))
            residuals = y - current  # negative gradient of 1/2 (y - F)^2
            tree = self._make_stage_tree(seed).fit(X, residuals)
            current += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            losses.append(mean_squared_error(y, current))

        self.loss_history_ = np.asarray(losses, dtype=float)
        self._mark_fitted()
        return self

    def _fit_classification(self, X: np.ndarray, y: np.ndarray):
        self._mapper = LabelMapper().fit(y)
        self.classes_ = self._mapper.classes_
        if self.classes_.size != 2:
            raise ValueError(
                f"ScratchGradientBoosting classification is binary; got "
                f"{self.classes_.size} classes: {self.classes_}."
            )
        target = self._mapper.transform(y).astype(float)

        base_rate = float(np.clip(np.mean(target), 1e-15, 1.0 - 1e-15))
        self.init_ = float(np.log(base_rate / (1.0 - base_rate)))
        current = np.full(X.shape[0], self.init_, dtype=float)
        rng = np.random.default_rng(self.random_state)
        eps = 1e-15
        losses = []
        self.estimators_ = []

        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, 2**31 - 1))
            probabilities = sigmoid(current)
            # Negative gradient of log-loss wrt the log-odds F: y - p.
            residuals = target - probabilities
            tree = self._make_stage_tree(seed).fit(X, residuals)
            current += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)

            p = np.clip(sigmoid(current), eps, 1.0 - eps)
            losses.append(
                -float(np.mean(target * np.log(p)
                               + (1.0 - target) * np.log(1.0 - p)))
            )

        self.loss_history_ = np.asarray(losses, dtype=float)
        self._mark_fitted()
        return self

    # ------------------------------------------------------------------
    def decision_function(self, X) -> np.ndarray:
        """Raw additive score ``F_M(X) = init_ + nu * sum_m h_m(X)``."""
        self._check_is_fitted()
        X = check_X(X)
        raw = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.estimators_:
            raw += self.learning_rate * tree.predict(X)
        return raw

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities of shape ``(n_samples, 2)`` (classification)."""
        self._check_is_fitted()
        if self._task != "classification":
            raise ValueError(
                "predict_proba is only defined for classification tasks."
            )
        p_positive = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p_positive, p_positive])

    def predict(self, X) -> np.ndarray:
        """Return labels (classification) or raw scores (regression)."""
        raw = self.decision_function(X)
        if self._task == "classification":
            positive = (raw >= 0.0).astype(int)
            return self._mapper.inverse_transform(positive)
        return raw


def demo() -> None:
    """Synthetic end-to-end checks for both ensembles."""
    print("=" * 62)
    print("ScratchRandomForest / ScratchGradientBoosting")
    print("=" * 62)

    # --- Random forest: classification on two moons --------------------
    X, y = make_moons(n_samples=500, noise=0.3, random_state=8)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=8)
    forest = ScratchRandomForest(
        n_estimators=25, max_depth=None, max_features="sqrt",
        random_state=8,
    ).fit(X_tr, y_tr)
    forest_train_acc = accuracy_score(y_tr, forest.predict(X_tr))
    forest_test_acc = accuracy_score(y_te, forest.predict(X_te))
    print("[RandomForest classification] 25 trees, bootstrap, max_features="
          "'sqrt'")
    print(f"  train accuracy: {forest_train_acc:.4f} | "
          f"test accuracy: {forest_test_acc:.4f}")

    # --- Random forest: regression -------------------------------------
    X, y = make_regression(n_samples=400, n_features=3, noise=0.5,
                           random_state=3)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=3)
    forest_reg = ScratchRandomForest(
        n_estimators=25, max_depth=8, max_features=None,
        task="regression", random_state=3,
    ).fit(X_tr, y_tr)
    print("[RandomForest regression] 25 trees, depth 8, all features")
    print(f"  test MSE: "
          f"{mean_squared_error(y_te, forest_reg.predict(X_te)):.4f} | "
          f"test R^2: {r2_score(y_te, forest_reg.predict(X_te)):.4f}")

    # --- Gradient boosting: non-linear regression -----------------------
    rng = np.random.default_rng(10)
    X = rng.uniform(-3.0, 3.0, size=(400, 1))
    y = np.sin(2.0 * X[:, 0]) + 0.2 * rng.normal(size=400)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=10)
    gbr = ScratchGradientBoosting(
        n_estimators=150, learning_rate=0.1, max_depth=3,
        task="regression", random_state=10,
    ).fit(X_tr, y_tr)
    print("[GradientBoosting regression] 150 stages, lr=0.1, depth=3")
    print(f"  train MSE: {mean_squared_error(y_tr, gbr.predict(X_tr)):.4f}"
          f" | test MSE: {mean_squared_error(y_te, gbr.predict(X_te)):.4f}"
          f" | test R^2: {r2_score(y_te, gbr.predict(X_te)):.4f}")
    print(f"  stage MSE: {gbr.loss_history_[0]:.4f} -> "
          f"{gbr.loss_history_[-1]:.4f}")

    # --- Gradient boosting: classification on two moons -----------------
    X, y = make_moons(n_samples=500, noise=0.3, random_state=11)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=11)
    gbc = ScratchGradientBoosting(
        n_estimators=150, learning_rate=0.1, max_depth=3,
        task="classification", random_state=11,
    ).fit(X_tr, y_tr)
    print("[GradientBoosting classification] 150 stages, lr=0.1, depth=3")
    print(f"  train accuracy: {accuracy_score(y_tr, gbc.predict(X_tr)):.4f}"
          f" | test accuracy: {accuracy_score(y_te, gbc.predict(X_te)):.4f}")
    print(f"  stage BCE: {gbc.loss_history_[0]:.4f} -> "
          f"{gbc.loss_history_[-1]:.4f}")


if __name__ == "__main__":
    demo()
