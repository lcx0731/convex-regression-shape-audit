"""Weighted univariate convex regression, with explicit numerical diagnostics."""

from .core import aggregate, fit, predict, slope_matrix

__all__ = ["aggregate", "fit", "predict", "slope_matrix"]
