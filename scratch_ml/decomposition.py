"""Principal Component Analysis implemented from scratch.

Math summary
------------
PCA finds the orthogonal directions of maximal variance of centered
data ``X_c = X - mean(X)``.

1. **Covariance route** (``method="eig"``): form the feature covariance
   matrix ``C = X_c^T X_c / (n - 1)`` and solve the eigenproblem

       ``C v = lambda v``

   with :func:`numpy.linalg.eigh` (the symmetric eigensolver behind
   ``numpy.linalg.eig`` -- real eigenvalues, orthonormal eigenvectors).
   Eigenvectors ordered by decreasing ``lambda`` are the principal
   components; ``lambda_i`` is the variance explained by component
   ``i``.

2. **SVD route** (``method="svd"``): ``X_c = U S V^T`` gives
   ``C = V (S^2 / (n-1)) V^T`` directly, so the columns of ``V`` are
   the same components and ``S^2 / (n - 1)`` the same variances --
   without forming ``C``.

Projection onto the first ``d`` components is the dimensionality
reduction ``Z = X_c V_d``, and the fraction of retained signal is

    ``explained_variance_ratio_i = lambda_i / sum_j lambda_j``.

Reconstruction ``X_hat = Z V_d^T + mean`` loses only the variances of
the dropped components.
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator
from .utils import check_X

__all__ = ["ScratchPCA"]


class ScratchPCA(BaseEstimator):
    """Linear dimensionality reduction via eigen-decomposition or SVD.

    Parameters
    ----------
    n_components : int or None
        Number of principal components to keep; ``None`` keeps all
        (``n_features``, or ``min(n_samples, n_features)`` for SVD).
    method : {"eig", "svd"}
        ``"eig"`` eigendecomposes the covariance matrix with
        ``np.linalg.eigh``; ``"svd"`` uses ``np.linalg.svd`` on the
        centered data.  Both yield the same components up to sign.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes, rows ordered by decreasing explained variance.
    explained_variance_ : ndarray of shape (n_components,)
        Variance captured by each component (``lambda_i``).
    explained_variance_ratio_ : ndarray of shape (n_components,)
        ``lambda_i`` divided by the total variance of the data.
    mean_ : ndarray of shape (n_features,)
        Per-feature mean subtracted before projection.
    """

    def __init__(self, n_components=None, method: str = "eig"):
        self.n_components = n_components
        self.method = method

    # ------------------------------------------------------------------
    def fit(self, X, y=None) -> "ScratchPCA":
        """Learn the principal axes of the centered data.

        ``y`` is ignored but accepted for the shared ``fit(X, y)``
        interface.

        Returns
        -------
        self
        """
        X = check_X(X)
        if self.method not in ("eig", "svd"):
            raise ValueError("method must be 'eig' or 'svd'.")

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("PCA needs at least 2 samples.")
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        if self.method == "eig":
            max_components = n_features
            if self.n_components is None:
                n_components = max_components
            else:
                n_components = int(self.n_components)
            if not 1 <= n_components <= max_components:
                raise ValueError(
                    f"n_components must be in [1, {max_components}], "
                    f"got {self.n_components}."
                )
            # Covariance matrix and its eigendecomposition.  eigh is
            # the symmetric-specialized form of np.linalg.eig: it
            # returns ascending real eigenvalues with orthonormal
            # eigenvectors, so we flip them into decreasing order.
            covariance = (X_centered.T @ X_centered) / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            self.components_ = eigenvectors[:, :n_components].T
            self.explained_variance_ = eigenvalues[:n_components]
            total_variance = float(np.sum(eigenvalues))
        else:
            # Thin SVD: X_c = U S V^T, components are rows of V.
            _, singular_values, Vt = np.linalg.svd(X_centered,
                                                   full_matrices=False)
            max_components = min(n_samples, n_features)
            if self.n_components is None:
                n_components = max_components
            else:
                n_components = int(self.n_components)
            if not 1 <= n_components <= max_components:
                raise ValueError(
                    f"n_components must be in [1, {max_components}], "
                    f"got {self.n_components}."
                )
            self.components_ = Vt[:n_components]
            variances = singular_values**2 / (n_samples - 1)
            self.explained_variance_ = variances[:n_components]
            total_variance = float(np.sum(variances))

        if total_variance > 0.0:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance
            )
        else:
            self.explained_variance_ratio_ = np.zeros_like(
                self.explained_variance_
            )

        self.n_components_ = n_components
        self._mark_fitted()
        return self

    # ------------------------------------------------------------------
    def transform(self, X) -> np.ndarray:
        """Project ``X`` onto the learned components: ``X_c @ V_d.T``."""
        self._check_is_fitted()
        X = check_X(X)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit the model and return the projection of ``X``."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, Z) -> np.ndarray:
        """Map projections back to the original space: ``Z @ V_d + mean``."""
        self._check_is_fitted()
        Z = check_X(Z)
        if Z.shape[1] != self.n_components_:
            raise ValueError(
                f"Expected {self.n_components_} components, "
                f"got {Z.shape[1]}."
            )
        return Z @ self.components_ + self.mean_


def demo() -> None:
    """Synthetic check: 3-D correlated data compressed to 2 components."""
    print("=" * 62)
    print("ScratchPCA")
    print("=" * 62)

    rng = np.random.default_rng(13)
    mean = [2.0, -1.0, 0.5]
    covariance = [[4.0, 2.0, 0.5], [2.0, 3.0, 0.2], [0.5, 0.2, 1.0]]
    X = rng.multivariate_normal(mean, covariance, size=400)

    pca = ScratchPCA(n_components=2, method="eig").fit(X)
    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)

    print("[PCA] method=eig, n_components=2 (from 3 features)")
    print(f"  explained variance ratio: "
          f"{np.round(pca.explained_variance_ratio_, 4)} | "
          f"cumulative: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  projection shape: {Z.shape} | reconstruction MSE: "
          f"{np.mean((X_hat - X) ** 2):.5f}")
    print(f"  reconstruction max abs error: "
          f"{np.max(np.abs(X_hat - X)):.5f}")

    # The SVD route must reproduce the same variances (up to sign).
    pca_svd = ScratchPCA(n_components=2, method="svd").fit(X)
    overlap = np.abs(pca.components_ @ pca_svd.components_.T)
    print("[PCA] method=svd cross-check")
    print(f"  explained variance ratio: "
          f"{np.round(pca_svd.explained_variance_ratio_, 4)}")
    print(f"  |components_eig . components_svd^T| (should be ~identity):\n"
          f"{np.round(overlap, 4)}")


if __name__ == "__main__":
    demo()
