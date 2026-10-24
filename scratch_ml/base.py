"""Base estimator plumbing that mirrors scikit-learn's API.

This module contains a minimal :class:`BaseEstimator` providing
``get_params`` / ``set_params`` / ``repr`` behaviour driven by the
``__init__`` signature, plus fitted-state checking helpers.  Every model
in :mod:`scratch_ml` inherits from it so that all estimators expose a
uniform, scikit-learn compatible interface.
"""

from __future__ import annotations

import inspect

__all__ = ["BaseEstimator", "NotFittedError"]


class NotFittedError(RuntimeError):
    """Raised when a predict/transform method is called before ``fit``."""


class BaseEstimator:
    """Minimal base class for every scratch estimator.

    Subclasses must assign every ``__init__`` argument to an attribute of
    the *same* name (the scikit-learn convention).  :meth:`get_params`
    then introspects the ``__init__`` signature, which keeps
    ``get_params`` / ``set_params`` working without boilerplate.
    """

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        """Return the estimator's constructor parameters.

        Parameters
        ----------
        deep : bool
            Kept for scikit-learn API compatibility; this implementation
            has no nested estimators, so it is ignored.

        Returns
        -------
        dict
            Mapping of parameter name -> current value.
        """
        signature = inspect.signature(type(self).__init__)
        return {
            name: getattr(self, name)
            for name in signature.parameters
            if name != "self"
        }

    def set_params(self, **params) -> "BaseEstimator":
        """Set parameters in place and return ``self`` (sklearn style)."""
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator "
                    f"{type(self).__name__}. Valid parameters are: "
                    f"{sorted(valid)}."
                )
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        items = sorted(self.get_params().items())
        params = ", ".join(f"{name}={value!r}" for name, value in items)
        return f"{type(self).__name__}({params})"

    # ------------------------------------------------------------------
    # Fitted-state bookkeeping
    # ------------------------------------------------------------------
    def _mark_fitted(self) -> None:
        """Record that ``fit`` completed successfully."""
        self._is_fitted = True

    def _check_is_fitted(self) -> None:
        """Raise :class:`NotFittedError` if ``fit`` has not run yet."""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                f"Call 'fit(X, y)' before using this method."
            )
