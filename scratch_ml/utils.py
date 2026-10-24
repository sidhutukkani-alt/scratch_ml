"""Shared validation helpers used by every estimator.

These functions intentionally mimic the small slice of
``sklearn.utils.validation`` that the estimators in this package need,
but they are written from scratch on top of NumPy only.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "check_X",
    "check_X_y",
    "check_random_state",
    "infer_task",
    "sigmoid",
    "train_test_split",
    "LabelMapper",
]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid ``1 / (1 + e^{-z})``.

    For ``z < 0`` the identity ``sigmoid(z) = e^{z} / (1 + e^{z})`` is
    used instead, which never evaluates ``e^{-z}`` for large positive
    ``-z`` (the overflow-prone branch).
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def check_X(X) -> np.ndarray:
    """Validate a feature matrix and return it as a 2-D float array.

    A 1-D input is interpreted as a single feature column, matching the
    scikit-learn convention.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got {X.ndim} dims.")
    if X.shape[0] == 0:
        raise ValueError("Feature matrix has 0 samples.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Feature matrix contains NaN or infinite values.")
    return X


def check_X_y(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Validate ``X`` and ``y`` together; ``y`` keeps its own dtype."""
    X = check_X(X)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    if y.ndim != 1:
        raise ValueError(f"Target vector must be 1-D, got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have inconsistent sample counts: "
            f"{X.shape[0]} vs {y.shape[0]}."
        )
    return X, y


def check_random_state(seed=None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` from a seed or ``None``."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def infer_task(y) -> str:
    """Infer ``"classification"`` or ``"regression"`` from the target.

    Float targets that contain non-integer values imply regression;
    integer-like, boolean or string targets imply classification.
    Ambiguous cases (e.g. integer counts used as regression targets)
    should be resolved by passing ``task=`` explicitly.
    """
    y = np.asarray(y)
    if y.dtype.kind == "f" and y.size and not np.all(np.mod(y, 1) == 0):
        return "regression"
    return "classification"


def train_test_split(
    X,
    y,
    test_size: float = 0.25,
    shuffle: bool = True,
    random_state=None,
):
    """Split ``X`` and ``y`` into train/test subsets.

    Parameters mirror scikit-learn: ``test_size`` is the fraction of
    samples (0 < ``test_size`` < 1) placed in the test split.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``.
    """
    X = check_X(X)
    X, y = check_X_y(X, y)
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

    n_samples = X.shape[0]
    n_test = max(1, int(round(n_samples * test_size)))
    if n_test >= n_samples:
        raise ValueError("test_size leaves no training samples.")

    if shuffle:
        order = check_random_state(random_state).permutation(n_samples)
        X, y = X[order], y[order]

    # Last ``n_test`` samples form the test split (sklearn convention).
    return X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]


class LabelMapper:
    """Bidirectional mapping between arbitrary class labels and ``0..K-1``.

    ``transform`` uses a binary search over the sorted unique labels
    produced by ``fit`` (i.e. ``np.searchsorted``), which works for
    strings, integers and floats alike.
    """

    def fit(self, y) -> "LabelMapper":
        """Learn the sorted set of distinct labels."""
        self.classes_ = np.unique(np.asarray(y))
        return self

    def transform(self, y) -> np.ndarray:
        """Map original labels to contiguous integer codes."""
        if not hasattr(self, "classes_"):
            raise RuntimeError("LabelMapper must be fitted before transform.")
        return np.searchsorted(self.classes_, np.asarray(y))

    def inverse_transform(self, codes) -> np.ndarray:
        """Map integer codes back to the original labels."""
        if not hasattr(self, "classes_"):
            raise RuntimeError(
                "LabelMapper must be fitted before inverse_transform."
            )
        return self.classes_[np.asarray(codes, dtype=int)]
