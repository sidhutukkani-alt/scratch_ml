"""Run every model's synthetic test suite.

Usage::

    python -m scratch_ml

Each module also keeps its own ``if __name__ == "__main__"`` block, so
a single model can be exercised with e.g.
``python -m scratch_ml.linear_model``.
"""

from . import (
    cluster,
    decomposition,
    ensemble,
    linear_model,
    naive_bayes,
    neighbors,
    svm,
    tree,
)

_MODULES = (
    linear_model,
    neighbors,
    naive_bayes,
    svm,
    tree,
    ensemble,
    cluster,
    decomposition,
)


def main() -> None:
    """Execute the demo of every algorithm module in order."""
    for module in _MODULES:
        module.demo()
        print()


if __name__ == "__main__":
    main()
