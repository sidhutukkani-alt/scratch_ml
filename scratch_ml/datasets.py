"""Synthetic dataset generators used by the ``__main__`` demos.

All generators return ``(X, y)`` pairs with ``X`` as a float64 matrix of
shape ``(n_samples, n_features)`` and ``y`` as a 1-D target vector, so
they plug directly into every estimator's ``fit(X, y)``.
"""

from __future__ import annotations

import numpy as np

from .utils import check_random_state

__all__ = [
    "make_regression",
    "make_classification",
    "make_blobs",
    "make_moons",
]


def make_regression(
    n_samples: int = 200,
    n_features: int = 3,
    noise: float = 0.1,
    bias: float = 0.0,
    random_state=None,
):
    """Linear regression problem ``y = X @ w + bias + N(0, noise^2)``.

    Returns
    -------
    (X, y) : tuple of numpy.ndarray
        ``X`` has shape ``(n_samples, n_features)``, ``y`` shape
        ``(n_samples,)``.
    """
    rng = check_random_state(random_state)
    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=n_features)
    y = X @ weights + bias + rng.normal(scale=noise, size=n_samples)
    return X, y.astype(float)


def make_classification(
    n_samples: int = 200,
    n_features: int = 2,
    n_classes: int = 2,
    class_sep: float = 2.0,
    class_std: float = 1.0,
    random_state=None,
):
    """Blob-style classification problem with Gaussian class clusters.

    Each class centre ``mu_c`` is drawn from ``N(0, class_sep^2)`` and
    each sample from ``N(mu_{y_i}, class_std^2)``.  Larger ``class_sep``
    (or smaller ``class_std``) makes the classes easier to separate.

    Returns
    -------
    (X, y) : tuple of numpy.ndarray
        Integer labels ``y`` lie in ``0 .. n_classes-1``.
    """
    rng = check_random_state(random_state)
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2.")
    centers = rng.normal(scale=class_sep, size=(n_classes, n_features))
    y = rng.integers(0, n_classes, size=n_samples)
    X = centers[y] + rng.normal(scale=class_std, size=(n_samples, n_features))
    return X.astype(float), y.astype(int)


def make_blobs(
    n_samples: int = 300,
    centers: int = 3,
    n_features: int = 2,
    cluster_std: float = 0.8,
    random_state=None,
):
    """Isotropic Gaussian blobs for clustering (KMeans demo).

    ``centers`` may be an int (centres drawn uniformly in ``[-10, 10]``)
    or an explicit ``(n_centers, n_features)`` array of centre
    coordinates.

    Returns
    -------
    (X, y) : tuple of numpy.ndarray
        ``y`` holds the ground-truth cluster index of each sample.
    """
    rng = check_random_state(random_state)
    if isinstance(centers, (int, np.integer)):
        n_centers = int(centers)
        center_coords = rng.uniform(-10.0, 10.0, size=(n_centers, n_features))
    else:
        center_coords = np.asarray(centers, dtype=float)
        n_centers = center_coords.shape[0]

    y = rng.integers(0, n_centers, size=n_samples)
    noise = rng.normal(scale=cluster_std, size=(n_samples, n_features))
    X = center_coords[y] + noise
    return X.astype(float), y.astype(int)


def make_moons(n_samples: int = 300, noise: float = 0.15, random_state=None):
    """Two interleaving half-circles (non-linearly separable).

    Returns
    -------
    (X, y) : tuple of numpy.ndarray
        ``y`` contains the binary labels 0 and 1.
    """
    rng = check_random_state(random_state)
    n_out = n_samples // 2
    n_in = n_samples - n_out

    t_outer = np.linspace(0.0, np.pi, n_out)
    outer = np.column_stack([np.cos(t_outer), np.sin(t_outer)])

    t_inner = np.linspace(0.0, np.pi, n_in)
    inner = np.column_stack(
        [1.0 - np.cos(t_inner), 1.0 - np.sin(t_inner) - 0.5]
    )

    X = np.vstack([outer, inner])
    y = np.concatenate([np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)])

    shuffle = rng.permutation(n_samples)
    X, y = X[shuffle], y[shuffle]
    X = X + rng.normal(scale=noise, size=X.shape)
    return X.astype(float), y
