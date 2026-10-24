"""K-Means clustering implemented from scratch.

Math summary
------------
K-Means partitions ``n`` samples into ``k`` sets ``C`` minimizing the
*within-cluster sum of squares* (inertia)

    ``W = sum_i ||x_i - mu_{a_i}||^2``

Lloyd's algorithm alternates the two optimal steps until convergence:

1. **Assignment** (given centers): ``a_i = argmin_k ||x_i - mu_k||^2``
2. **Update** (given assignments): ``mu_k = (1 / |C_k|) sum_{i in C_k} x_i``

Each iteration cannot increase ``W``, so the method converges to a
local minimum.  Because that minimum depends on initialization, the
run is repeated ``n_init`` times and the lowest-inertia solution wins.

**k-means++** seeding picks the first center uniformly at random and
each later center with probability proportional to ``D(x)^2`` (its
squared distance to the closest existing center), which provably keeps
the expected initial inertia within a constant factor of optimal.
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator
from .datasets import make_blobs
from .utils import check_X, check_random_state

__all__ = ["ScratchKMeans"]


class ScratchKMeans(BaseEstimator):
    """K-Means clustering with Lloyd iterations and k-means++ seeding.

    Parameters
    ----------
    n_clusters : int
        Number ``k`` of clusters to form.
    init : {"kmeans++", "random"}
        Centroid initialization strategy.
    n_init : int
        Number of independent restarts; the run with the lowest
        inertia is kept.
    max_iter : int
        Maximum Lloyd iterations per restart.
    tol : float
        Convergence threshold on the largest centroid shift
        (Euclidean norm) between two iterations.
    random_state : int or None
        Seed controlling initialization and restarts.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Final centroid coordinates.
    labels_ : ndarray of shape (n_samples,)
        Cluster index assigned to each *training* sample.
    inertia_ : float
        Sum of squared distances of samples to their assigned center.
    n_iter_ : int
        Lloyd iterations performed by the winning run.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "kmeans++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit(self, X, y=None) -> "ScratchKMeans":
        """Cluster ``X`` with repeated Lloyd runs and keep the best.

        ``y`` is ignored (K-Means is unsupervised) but accepted so the
        estimator follows the shared ``fit(X, y)`` interface.

        Returns
        -------
        self
        """
        X = check_X(X)
        n_samples = X.shape[0]
        if not 1 <= self.n_clusters <= n_samples:
            raise ValueError(
                f"n_clusters={self.n_clusters} must be between 1 and "
                f"n_samples={n_samples}."
            )
        if self.init not in ("kmeans++", "random"):
            raise ValueError("init must be 'kmeans++' or 'random'.")
        if self.n_init < 1:
            raise ValueError("n_init must be >= 1.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        rng = check_random_state(self.random_state)
        best = None
        for _ in range(self.n_init):
            candidate = self._run_lloyd(X, rng)
            if best is None or candidate[2] < best[2]:
                best = candidate

        self.labels_, self.cluster_centers_, self.inertia_, self.n_iter_ = best
        self._mark_fitted()
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """``(n_samples, k)`` table of ``||x - c||^2`` via the expansion
        ``||x||^2 - 2 x.c + ||c||^2`` (one BLAS matmul, no 3-D array)."""
        sq_norms = (
            np.sum(X**2, axis=1, keepdims=True)
            + np.sum(centers**2, axis=1)[None, :]
            - 2.0 * (X @ centers.T)
        )
        return np.maximum(sq_norms, 0.0)

    def _init_centers(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Seed centroids either uniformly at random or k-means++."""
        n_samples = X.shape[0]
        k = self.n_clusters
        if self.init == "random":
            chosen = rng.choice(n_samples, size=k, replace=False)
            return X[chosen].copy()

        centers = np.empty((k, X.shape[1]))
        centers[0] = X[rng.integers(n_samples)]
        closest = self._squared_distances(X, centers[:1]).ravel()
        for i in range(1, k):
            total = float(closest.sum())
            if total <= 0.0:
                idx = int(rng.integers(n_samples))  # all points are centers
            else:
                idx = int(rng.choice(n_samples, p=closest / total))
            centers[i] = X[idx]
            dist = self._squared_distances(X, centers[i:i + 1]).ravel()
            closest = np.minimum(closest, dist)
        return centers

    def _run_lloyd(self, X: np.ndarray, rng: np.random.Generator):
        """One seeded run; returns ``(labels, centers, inertia, n_iter)``."""
        n_samples = X.shape[0]
        centers = self._init_centers(X, rng)
        labels = np.zeros(n_samples, dtype=int)
        n_iter = 0

        for n_iter in range(1, self.max_iter + 1):
            # 1) Assignment step: nearest centroid per sample.
            labels = np.argmin(
                self._squared_distances(X, centers), axis=1
            )
            # 2) Update step: mean of assigned points (empty clusters
            #    keep their previous centroid instead of becoming NaN).
            new_centers = np.array(
                [
                    X[labels == k].mean(axis=0)
                    if np.any(labels == k)
                    else centers[k]
                    for k in range(self.n_clusters)
                ]
            )
            shift = float(
                np.max(np.linalg.norm(new_centers - centers, axis=1))
            )
            centers = new_centers
            if shift <= self.tol:
                break

        distances = self._squared_distances(X, centers)
        labels = np.argmin(distances, axis=1)
        inertia = float(
            np.sum(distances[np.arange(n_samples), labels])
        )
        return labels, centers, inertia, n_iter

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """Assign each row of ``X`` to its nearest fitted centroid."""
        self._check_is_fitted()
        X = check_X(X)
        return np.argmin(self._squared_distances(X, self.cluster_centers_),
                         axis=1)

    def transform(self, X) -> np.ndarray:
        """Euclidean distance from every sample to every centroid."""
        self._check_is_fitted()
        X = check_X(X)
        return np.sqrt(self._squared_distances(X, self.cluster_centers_))

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit the model and return the distance embedding of ``X``."""
        return self.fit(X, y).transform(X)


def demo() -> None:
    """Synthetic clustering check: recover four Gaussian blobs."""
    print("=" * 62)
    print("ScratchKMeans")
    print("=" * 62)

    true_centers = np.array(
        [[-5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, 5.0]]
    )
    X, y = make_blobs(n_samples=400, centers=true_centers, cluster_std=1.0,
                      random_state=12)
    model = ScratchKMeans(
        n_clusters=4, init="kmeans++", n_init=10, random_state=12
    ).fit(X)

    # Cluster labels are arbitrary, so score by purity: for each
    # predicted cluster take its majority ground-truth blob.
    correct = 0
    for cluster in range(model.n_clusters):
        members = y[model.labels_ == cluster]
        if members.size:
            correct += int(np.bincount(members).max())
    purity = correct / X.shape[0]

    print(f"  inertia: {model.inertia_:.4f} | iterations: {model.n_iter_} "
          f"| k: {model.n_clusters}")
    print(f"  cluster purity vs ground truth: {purity:.4f}")
    print(f"  recovered centers:\n{np.round(model.cluster_centers_, 2)}")
    print(f"  true centers:\n{true_centers}")

    # ``transform`` gives distances, ``predict`` re-assigns new points.
    distances = model.transform(X[:3])
    print(f"  first 3 samples assigned to: {model.predict(X[:3]).tolist()} "
          f"(distance shapes {distances.shape})")


if __name__ == "__main__":
    demo()
