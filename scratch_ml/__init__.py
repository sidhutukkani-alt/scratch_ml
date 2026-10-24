"""scratch_ml: classical ML algorithms implemented from scratch.

Every estimator in this package is written with **only NumPy and the
standard library** -- no scikit-learn, SciPy or other ML frameworks --
while mimicking the scikit-learn API (``fit(X, y)`` / ``predict(X)`` /
``predict_proba`` plus fitted attributes ending in ``_``).

Modules
-------
``linear_model``    : ScratchLinearRegression, ScratchLogisticRegression
``neighbors``       : ScratchKNN
``naive_bayes``     : ScratchNaiveBayes (Gaussian)
``svm``             : ScratchLinearSVM (hinge loss + gradient descent)
``tree``            : ScratchDecisionTree (gini / entropy / mse)
``ensemble``        : ScratchRandomForest, ScratchGradientBoosting
``cluster``         : ScratchKMeans
``decomposition``   : ScratchPCA
``metrics``         : accuracy / MSE / MAE / R^2 helpers
``datasets``        : synthetic data generators for the demos

Run every model's synthetic test with::

    python -m scratch_ml

or a single one with e.g. ``python -m scratch_ml.tree`` (package
modules use relative imports, so always run them via ``-m`` from the
project root).
"""

from .base import BaseEstimator, NotFittedError
from .cluster import ScratchKMeans
from .datasets import (
    make_blobs,
    make_classification,
    make_moons,
    make_regression,
)
from .decomposition import ScratchPCA
from .ensemble import ScratchGradientBoosting, ScratchRandomForest
from .linear_model import ScratchLinearRegression, ScratchLogisticRegression
from .metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from .naive_bayes import ScratchNaiveBayes
from .neighbors import ScratchKNN
from .svm import ScratchLinearSVM
from .tree import ScratchDecisionTree
from .utils import train_test_split

__version__ = "0.1.0"

__all__ = [
    # Base API
    "BaseEstimator",
    "NotFittedError",
    # Supervised models
    "ScratchLinearRegression",
    "ScratchLogisticRegression",
    "ScratchKNN",
    "ScratchNaiveBayes",
    "ScratchLinearSVM",
    "ScratchDecisionTree",
    "ScratchRandomForest",
    "ScratchGradientBoosting",
    # Unsupervised models
    "ScratchKMeans",
    "ScratchPCA",
    # Utilities
    "accuracy_score",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
    "train_test_split",
    "make_blobs",
    "make_classification",
    "make_moons",
    "make_regression",
]
