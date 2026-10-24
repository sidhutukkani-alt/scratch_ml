"""K-Nearest Neighbors implemented from scratch.

Math summary
------------
KNN is a *lazy* learner: ``fit`` only stores the training set.  For a
query point ``q`` the algorithm

1. computes the distance to every training point, either

   * Euclidean:   ``d(q, x) = sqrt(sum_j (q_j - x_j)^2)``
   * Manhattan:   ``d(q, x) = sum_j |q_j - x_j|``
2. selects the ``k`` smallest distances,
3. predicts

   * classification -- the *majority vote* among the ``k`` neighbours
     (ties resolved in favour of the smaller class label), or
   * regression -- the *mean* of the ``k`` neighbour targets.
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator
from .datasets import make_blobs, make_regression
from .metrics import accuracy_score, mean_squared_error, r2_score
from .utils import (
    LabelMapper,
    check_X,
    check_X_y,
    infer_task,
    train_test_split,
)

__all__ = ["ScratchKNN"]

_METRICS = ("euclidean", "manhattan")


class ScratchKNN(BaseEstimator):
    """K-Nearest Neighbors for classification and regression.

    Parameters
    ----------
    n_neighbors : int
        Number ``k`` of neighbors considered at prediction time.
    metric : {"euclidean", "manhattan"}
        Distance function used to rank training points.
    task : {"auto", "classification", "regression"}
        ``"auto"`` inspects ``y`` (non-integer float targets imply
        regression) at :meth:`fit` time.

    Attributes
    ----------
    classes_ : ndarray
        Sorted class labels (classification only).
    n_train_ : int
        Number of stored training samples.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        task: str = "auto",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.task = task

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "ScratchKNN":
        """Store the training set (KNN performs no optimization).

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1.")
        if self.metric not in _METRICS:
            raise ValueError(
                f"metric must be one of {_METRICS}, got {self.metric!r}."
            )
        if self.task not in ("auto", "classification", "regression"):
            raise ValueError(
                "task must be 'auto', 'classification' or 'regression'."
            )

        self._task = infer_task(y) if self.task == "auto" else self.task
        if self._task == "classification":
            self._mapper = LabelMapper().fit(y)
            self.classes_ = self._mapper.classes_
        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} exceeds the number of "
                f"training samples ({X.shape[0]})."
            )

        self._X_train = X
        self._y_train = y
        self.n_train_ = X.shape[0]
        self._mark_fitted()
        return self

    # ------------------------------------------------------------------
    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Find the ``k`` nearest training indices for each query point.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features) or None
            Query points.  ``None`` queries the training set itself and
            excludes each sample's own index (sklearn convention).
        n_neighbors : int or None
            Defaults to :attr:`n_neighbors`.
        return_distance : bool
            If ``True`` return ``(distances, indices)``, else indices.

        Returns
        -------
        distances : ndarray of shape (n_queries, k)
            Sorted neighbor distances (only if ``return_distance``).
        indices : ndarray of shape (n_queries, k)
            Indices into the training set.
        """
        self._check_is_fitted()
        k = self.n_neighbors if n_neighbors is None else int(n_neighbors)
        if k < 1:
            raise ValueError("n_neighbors must be >= 1.")

        if X is None:
            query = self._X_train
            k = min(k, self.n_train_ - 1) if self.n_train_ > 1 else 1
            distances = self._pairwise_distances(query, self._X_train)
            # Drop self-matches so a point is never its own neighbor.
            own = np.arange(self.n_train_)
            distances[own, own] = np.inf
        else:
            query = check_X(X)
            k = min(k, self.n_train_)
            distances = self._pairwise_distances(query, self._X_train)

        if k > distances.shape[1]:
            raise ValueError(
                f"n_neighbors={k} exceeds the number of training samples "
                f"({self.n_train_})."
            )
        indices = np.argsort(distances, axis=1, kind="mergesort")[:, :k]
        if return_distance:
            row = np.arange(indices.shape[0])[:, None]
            return distances[row, indices], indices
        return indices

    def _pairwise_distances(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Dense ``(n_A, n_B)`` distance matrix for the chosen metric."""
        diff = A[:, None, :] - B[None, :, :]
        if self.metric == "euclidean":
            return np.sqrt(np.maximum(np.sum(diff**2, axis=-1), 0.0))
        return np.sum(np.abs(diff), axis=-1)

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """Majority vote (classification) or neighbor mean (regression)."""
        self._check_is_fitted()
        _, indices = self.kneighbors(X, return_distance=True)
        neighbor_targets = self._y_train[indices]  # (n_queries, k)

        if self._task == "regression":
            return neighbor_targets.mean(axis=1).astype(float)

        # Majority vote: one-hot encode neighbor labels and count.
        codes = self._mapper.transform(neighbor_targets)
        votes = np.eye(self.classes_.size, dtype=float)[codes].sum(axis=1)
        # argmax breaks ties towards the smallest class code.
        return self._mapper.inverse_transform(np.argmax(votes, axis=1))


def demo() -> None:
    """Synthetic end-to-end checks for classification and regression."""
    print("=" * 62)
    print("ScratchKNN")
    print("=" * 62)

    # --- Classification on 3 overlapping Gaussian blobs ----------------
    centers = np.array([[0.0, 0.0], [4.0, 1.0], [1.0, 4.0]])
    X, y = make_blobs(n_samples=450, centers=centers, cluster_std=1.1,
                      random_state=2)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=2)
    for metric in ("euclidean", "manhattan"):
        clf = ScratchKNN(n_neighbors=7, metric=metric,
                         task="classification").fit(X_tr, y_tr)
        train_acc = accuracy_score(y_tr, clf.predict(X_tr))
        test_acc = accuracy_score(y_te, clf.predict(X_te))
        print(f"[KNN classification, metric={metric}] k=7")
        print(f"  train accuracy: {train_acc:.4f} | "
              f"test accuracy: {test_acc:.4f}")

    # --- Regression on a linear target --------------------------------
    X, y = make_regression(n_samples=300, n_features=3, noise=0.5,
                           random_state=3)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=3)
    reg = ScratchKNN(n_neighbors=5, task="regression").fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    print("[KNN regression] k=5, euclidean")
    print(f"  test MSE: {mean_squared_error(y_te, pred):.4f} | "
          f"test R^2: {r2_score(y_te, pred):.4f}")


if __name__ == "__main__":
    demo()
