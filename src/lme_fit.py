"""One fitting rule for every linear mixed model in the pipeline.

REML with L-BFGS first. If L-BFGS does not converge, refit with Powell and keep
the Powell fit only if it converged and its log-likelihood is not lower than the
L-BFGS one. When L-BFGS converges both reach the same maximum, so Powell (slower,
gradient-free) is only a fallback for flat or irregular likelihood surfaces.
"""
from __future__ import annotations

from typing import Tuple

MAXITER = 2000


def fit_lbfgs_powell(model, maxiter: int = MAXITER) -> Tuple[object, str]:
    """Fit a statsmodels MixedLM model; return (result, optimiser label)."""
    fit = model.fit(reml=True, method="lbfgs", maxiter=maxiter)
    if bool(getattr(fit, "converged", True)):
        return fit, "lbfgs"
    refit = model.fit(reml=True, method="powell", maxiter=maxiter)
    if bool(getattr(refit, "converged", True)) and refit.llf >= fit.llf:
        return refit, "powell"
    return fit, "lbfgs (not converged)"
