"""Decision trees (CART) implemented from scratch.

Math summary
------------
A tree is grown by **recursive binary splitting**.  At every node we
search over all features and candidate thresholds (the midpoints
between consecutive sorted feature values) for the split maximizing
the impurity *decrease*

    ``Delta = I(parent) - (n_left / n) * I(left)
                           - (n_right / n) * I(right)``

where ``I`` is one of:

* **Gini impurity** (classification):
  ``I = 1 - sum_c p_c^2``  with ``p_c`` the class proportions,
* **Entropy / information gain** (classification):
  ``I = -sum_c p_c log2(p_c)``, so ``Delta`` is the *information gain*,
* **MSE / variance reduction** (regression):
  ``I = (1/n) sum_i (y_i - y_bar)^2`` (equivalently the residual sum
  of squares ``SSE = sum y_i^2 - (sum y_i)^2 / n``).

Split evaluation is vectorized with cumulative class counts (resp.
cumulative sums) along the sorted feature axis, so each feature costs
``O(n log n)`` for sorting instead of ``O(n * n_thresholds)``.

Recursion stops at ``max_depth``, when a node holds fewer than
``min_samples_split`` samples, when the node is pure, or when no split
improves the impurity (respecting ``min_samples_leaf``).
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator
from .datasets import make_regression, make_moons
from .metrics import accuracy_score, mean_squared_error
from .utils import (
    LabelMapper,
    check_X,
    check_X_y,
    infer_task,
    train_test_split,
)

__all__ = ["DecisionNode", "ScratchDecisionTree"]

_CLASSIFICATION_CRITERIA = ("gini", "entropy")
_REGRESSION_CRITERIA = ("mse",)


class DecisionNode:
    """A node of a binary decision tree.

    Internal nodes hold the splitting rule ``x[feature] <= threshold``
    plus ``left``/``right`` children (``value is None``).  Leaf nodes
    hold only ``value``: the class-probability vector for
    classification trees or the mean target for regression trees.
    """

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        n_samples: int = 0,
        impurity: float = 0.0,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.impurity = impurity

    @property
    def is_leaf(self) -> bool:
        """``True`` when the node has no children."""
        return self.left is None and self.right is None

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        if self.is_leaf:
            return f"DecisionNode(leaf, n={self.n_samples})"
        return (
            f"DecisionNode(x[{self.feature}] <= {self.threshold:.4f}, "
            f"n={self.n_samples})"
        )


# ----------------------------------------------------------------------
# Vectorized impurity helpers (rows of ``counts`` = class histograms)
# ----------------------------------------------------------------------
def _impurity(counts: np.ndarray, criterion: str) -> np.ndarray:
    """Gini or entropy for each row of a ``(m, n_classes)`` count table."""
    total = counts.sum(axis=1, keepdims=True)
    p = np.divide(
        counts, total, out=np.zeros_like(counts, dtype=float), where=total > 0
    )
    if criterion == "gini":
        return 1.0 - np.sum(p * p, axis=1)
    log_p = np.zeros_like(p)
    np.log2(p, out=log_p, where=p > 0)  # 0 * log 0 := 0
    return -np.sum(p * log_p, axis=1)


def _class_counts(y_enc: np.ndarray, n_classes: int) -> np.ndarray:
    """Class histogram of integer-encoded labels."""
    return np.bincount(y_enc, minlength=n_classes).astype(float)


class ScratchDecisionTree(BaseEstimator):
    """CART-style decision tree for classification and regression.

    Parameters
    ----------
    criterion : {"gini", "entropy", "mse"} or None
        Impurity function to minimize.  ``None`` picks the task
        default: ``"gini"`` for classification, ``"mse"``
        (variance reduction) for regression.
    max_depth : int or None
        Maximum tree depth; ``None`` grows until other stops apply.
    min_samples_split : int
        A node needs at least this many samples to be split.
    min_samples_leaf : int
        Both children of a split must contain at least this many
        samples.
    max_features : {"sqrt", "log2", int, float, None}
        Number of candidate features examined at *each* node.  ``None``
        uses all features; strings/ints/floats in ``(0, 1]`` draw a
        random subset (this is what makes the random forest random).
    task : {"auto", "classification", "regression"}
        ``"auto"`` infers the task from ``y``.
    random_state : int or None
        Seed for the per-node feature subset draws.

    Attributes
    ----------
    tree_ : DecisionNode
        Root node of the fitted tree.
    classes_ : ndarray
        Sorted class labels (classification only).
    n_nodes_ : int
        Total number of nodes (leaves included) in the fitted tree.
    """

    def __init__(
        self,
        criterion=None,
        max_depth=None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features=None,
        task: str = "auto",
        random_state=None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.task = task
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "ScratchDecisionTree":
        """Grow the tree by recursive binary splitting.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        if self.task not in ("auto", "classification", "regression"):
            raise ValueError(
                "task must be 'auto', 'classification' or 'regression'."
            )
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be >= 1 or None.")

        self._task = infer_task(y) if self.task == "auto" else self.task
        if self._task == "classification":
            self._criterion = (
                "gini" if self.criterion is None else self.criterion
            )
            if self._criterion not in _CLASSIFICATION_CRITERIA:
                raise ValueError(
                    f"criterion {self._criterion!r} is not a classification "
                    f"criterion; use one of {_CLASSIFICATION_CRITERIA}."
                )
            self._mapper = LabelMapper().fit(y)
            self.classes_ = self._mapper.classes_
            self.n_classes_ = self.classes_.size
            target = self._mapper.transform(y).astype(int)
        else:
            self._criterion = (
                "mse" if self.criterion is None else self.criterion
            )
            if self._criterion not in _REGRESSION_CRITERIA:
                raise ValueError(
                    f"criterion {self._criterion!r} is not a regression "
                    f"criterion; use one of {_REGRESSION_CRITERIA}."
                )
            target = y.astype(float)

        self._n_features = X.shape[1]
        self._max_features_count = self._resolve_max_features()
        self._rng = np.random.default_rng(self.random_state)

        self.tree_ = self._build(X, target, depth=0)
        self.n_nodes_ = self._count_nodes(self.tree_)
        self._mark_fitted()
        return self

    # ------------------------------------------------------------------
    def _resolve_max_features(self):
        """Translate the ``max_features`` argument into a count or None."""
        p = self._n_features
        mf = self.max_features
        if mf is None:
            return None
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(round(np.sqrt(p))))
            if mf == "log2":
                return max(1, int(np.floor(np.log2(max(p, 2)))))
            raise ValueError(f"Unknown max_features value {mf!r}.")
        if isinstance(mf, float):
            if not 0.0 < mf <= 1.0:
                raise ValueError("Fractional max_features must be in (0, 1].")
            return max(1, int(round(mf * p)))
        count = int(mf)
        if count < 1:
            raise ValueError("max_features must be >= 1.")
        return min(count, p)

    def _feature_subset(self) -> np.ndarray:
        """Random subset (or all) of feature indices for one node."""
        p = self._n_features
        k = self._max_features_count
        if k is None or k >= p:
            return np.arange(p)
        return np.sort(self._rng.choice(p, size=k, replace=False))

    # ------------------------------------------------------------------
    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionNode:
        """Recursively grow the subtree covering the given samples."""
        n_samples = X.shape[0]
        if self._task == "classification":
            counts = _class_counts(y, self.n_classes_)
            impurity = float(_impurity(counts[None, :], self._criterion)[0])
            value = counts / counts.sum()
        else:
            impurity = float(np.var(y))
            value = float(np.mean(y))

        node = DecisionNode(
            value=value, n_samples=n_samples, impurity=impurity
        )

        # --- stopping conditions -------------------------------------
        pure = impurity <= 1e-12
        depth_limited = self.max_depth is not None and depth >= self.max_depth
        too_small = n_samples < self.min_samples_split
        if pure or depth_limited or too_small:
            return node

        split = self._best_split(X, y)
        if split is None:
            return node  # no impurity decrease -> keep as leaf

        feature, threshold = split
        left_mask = X[:, feature] <= threshold
        if left_mask.sum() < self.min_samples_leaf:
            return node

        node.feature = int(feature)
        node.threshold = float(threshold)
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        node.value = None  # internal nodes predict through their leaves
        return node

    # ------------------------------------------------------------------
    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Best ``(feature, threshold)`` or ``None`` if no gain."""
        if self._task == "classification":
            return self._best_split_classification(X, y)
        return self._best_split_regression(X, y)

    def _best_split_classification(self, X: np.ndarray, y: np.ndarray):
        """Maximize information gain using cumulative class counts.

        Class counts are accumulated along the feature-sorted axis, so
        every candidate threshold is evaluated with two vectorized
        slices instead of a Python loop over thresholds.
        """
        n_samples = X.shape[0]
        total_counts = _class_counts(y, self.n_classes_)
        parent_counts = total_counts[None, :]
        parent_imp = float(_impurity(parent_counts, self._criterion)[0])
        one_hot = np.eye(self.n_classes_, dtype=float)[y]

        best_gain = 1e-12
        best_split = None
        for feature in self._feature_subset():
            values = X[:, feature]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            if sorted_values[-1] == sorted_values[0]:
                continue  # constant feature: no candidate threshold

            # Positions where the sorted value changes: each one is a
            # candidate split (threshold = midpoint of the two values).
            boundaries = np.nonzero(np.diff(sorted_values))[0]
            cum_counts = np.cumsum(one_hot[order], axis=0)
            left_counts = cum_counts[boundaries]  # (m, n_classes)
            n_left = (boundaries + 1).astype(float)
            n_right = n_samples - n_left
            right_counts = total_counts - left_counts

            valid = (n_left >= self.min_samples_leaf) & (
                n_right >= self.min_samples_leaf
            )
            if not np.any(valid):
                continue

            left_imp = _impurity(left_counts, self._criterion)
            right_imp = _impurity(right_counts, self._criterion)
            weighted = (n_left * left_imp + n_right * right_imp) / n_samples
            gains = parent_imp - weighted
            gains[~valid] = -np.inf

            j = int(np.argmax(gains))
            if gains[j] > best_gain:
                best_gain = float(gains[j])
                threshold = 0.5 * (
                    sorted_values[boundaries[j]]
                    + sorted_values[boundaries[j] + 1]
                )
                best_split = (int(feature), float(threshold))
        return best_split

    def _best_split_regression(self, X: np.ndarray, y: np.ndarray):
        """Minimize the residual sum of squares using cumulative sums.

        For a sorted feature, ``SSE = sum(y^2) - (sum y)^2 / n`` holds
        for each side of a candidate split, so cumulative sums of ``y``
        and ``y^2`` give every candidate's SSE in closed form.
        """
        n_samples = X.shape[0]
        total_sum = float(np.sum(y))
        total_sq = float(np.sum(y * y))
        parent_sse = max(total_sq - total_sum**2 / n_samples, 0.0)

        best_sse = parent_sse - 1e-12  # require a strict decrease
        best_split = None
        for feature in self._feature_subset():
            values = X[:, feature]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            if sorted_values[-1] == sorted_values[0]:
                continue  # constant feature

            boundaries = np.nonzero(np.diff(sorted_values))[0]
            sorted_y = y[order]
            cum_sum = np.cumsum(sorted_y)
            cum_sq = np.cumsum(sorted_y**2)

            s_left = cum_sum[boundaries]
            sq_left = cum_sq[boundaries]
            n_left = (boundaries + 1).astype(float)
            n_right = n_samples - n_left
            s_right = total_sum - s_left
            sq_right = total_sq - sq_left

            valid = (n_left >= self.min_samples_leaf) & (
                n_right >= self.min_samples_leaf
            )
            if not np.any(valid):
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                sse = (sq_left - s_left**2 / n_left) + (
                    sq_right - s_right**2 / n_right
                )
            sse = np.maximum(sse, 0.0)
            sse[~valid] = np.inf

            j = int(np.argmin(sse))
            if sse[j] < best_sse:
                best_sse = float(sse[j])
                threshold = 0.5 * (
                    sorted_values[boundaries[j]]
                    + sorted_values[boundaries[j] + 1]
                )
                best_split = (int(feature), float(threshold))
        return best_split

    # ------------------------------------------------------------------
    def _traverse(self, x: np.ndarray) -> DecisionNode:
        """Walk the tree from the root to the leaf governing sample ``x``."""
        node = self.tree_
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    @staticmethod
    def _count_nodes(node: DecisionNode) -> int:
        if node.is_leaf:
            return 1
        return 1 + ScratchDecisionTree._count_nodes(
            node.left
        ) + ScratchDecisionTree._count_nodes(node.right)

    def predict_proba(self, X) -> np.ndarray:
        """Leaf class-probability vectors of shape ``(n_samples, K)``."""
        self._check_is_fitted()
        if self._task != "classification":
            raise ValueError("predict_proba is only defined for classifiers.")
        leaves = [self._traverse(row) for row in check_X(X)]
        return np.array([leaf.value for leaf in leaves], dtype=float)

    def predict(self, X) -> np.ndarray:
        """Predict class labels (classification) or means (regression)."""
        self._check_is_fitted()
        leaves = [self._traverse(row) for row in check_X(X)]
        if self._task == "classification":
            proba = np.array([leaf.value for leaf in leaves], dtype=float)
            return self._mapper.inverse_transform(np.argmax(proba, axis=1))
        return np.array([leaf.value for leaf in leaves], dtype=float)


def demo() -> None:
    """Synthetic end-to-end checks for classification and regression."""
    print("=" * 62)
    print("ScratchDecisionTree")
    print("=" * 62)

    # --- Classification on two moons (non-linear boundary) ------------
    X, y = make_moons(n_samples=500, noise=0.25, random_state=6)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=6)
    for criterion in ("gini", "entropy"):
        clf = ScratchDecisionTree(
            criterion=criterion, max_depth=8, random_state=6
        ).fit(X_tr, y_tr)
        train_acc = accuracy_score(y_tr, clf.predict(X_tr))
        test_acc = accuracy_score(y_te, clf.predict(X_te))
        print(f"[DecisionTree classification, criterion={criterion}] depth=8")
        print(f"  train accuracy: {train_acc:.4f} | "
              f"test accuracy: {test_acc:.4f} | "
              f"nodes: {clf.n_nodes_}")

    # --- Regression (variance reduction / MSE splits) ------------------
    X, y = make_regression(n_samples=400, n_features=3, noise=0.5,
                           random_state=7)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=7)
    reg = ScratchDecisionTree(criterion="mse", max_depth=6,
                              random_state=7).fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    print("[DecisionTree regression, criterion=mse] depth=6")
    print(f"  train MSE: {mean_squared_error(y_tr, reg.predict(X_tr)):.4f}"
          f" | test MSE: {mean_squared_error(y_te, pred):.4f}"
          f" | nodes: {reg.n_nodes_}")


if __name__ == "__main__":
    demo()
