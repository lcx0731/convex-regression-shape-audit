"""Original small active-set implementation for weighted convex least squares.

Only NumPy is needed by the core. SciPy and exact rational face enumeration
are independent verification paths, not implementation dependencies.
"""

import math
from numbers import Real

import numpy as np


def _vector(values, name, maximum=500, bound=1e9):
    if not isinstance(values, (list, tuple, np.ndarray)) or np.ndim(values) != 1 or not 1 <= len(values) <= maximum:
        raise ValueError(f"{name} must be a one-dimensional vector of length 1 to {maximum}")
    converted = []
    for v in values:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, Real):
            raise ValueError(f"{name} must contain real numbers, not booleans")
        try:
            value = float(v)
        except OverflowError as exc:
            raise ValueError(f"{name} contains an oversized number") from exc
        if not math.isfinite(value) or abs(value) > bound:
            raise ValueError(f"{name} must be finite with absolute values at most {bound}")
        converted.append(value)
    return np.asarray(converted)


def aggregate(x, y, weights=None):
    """Combine exactly equal design values; retain within-group weighted SSE."""
    x, y = _vector(x, "x"), _vector(y, "y", bound=1e6)
    weights = np.ones(len(x)) if weights is None else _vector(weights, "weights", bound=1e6)
    if len(x) != len(y) or len(x) != len(weights) or np.any(weights < 1e-12):
        raise ValueError("x, y, and weights >= 1e-12 must have equal lengths")
    if np.max(weights)/np.min(weights) > 1e6:
        raise ValueError("weight ratio exceeds 1e6")
    groups = {}
    for a, b, w in zip(x, y, weights):
        groups.setdefault(float(a), []).append((float(b), float(w)))
    grid, means, totals = [], [], []
    within = []
    for a, items in sorted(groups.items()):
        total = math.fsum(w for _, w in items)
        mean = math.fsum(b*w for b, w in items)/total
        grid.append(a)
        means.append(mean)
        totals.append(total)
        within.extend(w*(b-mean)**2 for b, w in items)
    if not 2 <= len(grid) <= 80:
        raise ValueError("need 2 to 80 distinct design values")
    grid = np.array(grid)
    if grid[-1]-grid[0] < 1e-12:
        raise ValueError("design span must be at least 1e-12")
    if np.min(np.diff(grid))/(grid[-1]-grid[0]) < 1e-7:
        raise ValueError("distinct design values are too close relative to their span")
    if max(totals)/min(totals) > 1e6:
        raise ValueError("aggregated weight ratio exceeds 1e6")
    return {"x": grid, "y": np.array(means), "weights": np.array(totals),
            "within_group_sse": math.fsum(within), "observation_count": len(x)}


def slope_matrix(x):
    """D @ f is the vector of adjacent slope increases on a sorted grid."""
    x = _vector(x, "x", maximum=80)
    if len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("x must have at least two strictly increasing values")
    gaps = np.diff(x)
    if x[-1]-x[0] < 1e-12 or np.min(gaps)/(x[-1]-x[0]) < 1e-7:
        raise ValueError("design span or spacing is too small")
    d = np.zeros((max(0, len(x)-2), len(x)))
    for j in range(len(x)-2):
        d[j, j:j+3] = [1/gaps[j], -1/gaps[j]-1/gaps[j+1], 1/gaps[j+1]]
    return d


def _active_nnls(a, b, max_steps, tolerance=1e-10):
    """Solve min ||a q - b||^2 for q>=0 by passive-set least squares.

    Each reduced least-squares solve consumes one step. Boundary moves preserve
    nonnegativity. A budget exit returns a candidate, never claims convergence.
    """
    m = a.shape[1]
    q = np.zeros(m)
    passive = np.zeros(m, dtype=bool)
    solves = additions = removals = 0
    while True:
        gradient = a.T @ (b-a @ q)
        candidates = np.flatnonzero(~passive & (gradient > tolerance))
        if not len(candidates):
            return q, "stationary", {"least_squares_solves": solves, "additions": additions, "removals": removals}
        j = candidates[np.argmax(gradient[candidates])]
        passive[j] = True
        additions += 1
        while True:
            if solves >= max_steps:
                return q, "iteration_limit", {"least_squares_solves": solves, "additions": additions, "removals": removals}
            trial = np.zeros(m)
            trial[passive] = np.linalg.lstsq(a[:, passive], b, rcond=1e-12)[0]
            solves += 1
            bad = np.flatnonzero(passive & (trial <= 0))
            if not len(bad):
                q = trial
                break
            denom = q[bad]-trial[bad]
            ratios = np.divide(q[bad], denom, out=np.zeros(len(bad)), where=denom > 0)
            hit = bad[np.argmin(ratios)]
            fraction = float(np.min(ratios))
            q += fraction*(trial-q)
            q[hit] = 0.0
            passive[hit] = False
            # Clear any additional roundoff-sized negatives created by the move.
            negative = q < 0
            passive[negative] = False
            q[negative] = 0.0
            removals += 1 + int(np.sum(negative))


def fit(x, y, weights=None, *, max_steps=10000):
    """Fit weighted convex regression and return JSON-compatible diagnostics.

    Floating-point feasibility and KKT checks are numerical diagnostics, not
    rigorous certificates. Only status='converged' should be used to predict.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 10000:
        raise ValueError("max_steps must be an integer from 1 to 10000")
    grouped = aggregate(x, y, weights)
    grid, means, raw_weights = grouped["x"], grouped["y"], grouped["weights"]
    unit_x = (grid-grid[0])/(grid[-1]-grid[0])
    w_scale = float(max(raw_weights))
    w = raw_weights/w_scale
    root_w = np.sqrt(w)
    affine = np.column_stack([np.ones(len(grid)), unit_x])
    trend = affine @ np.linalg.lstsq(root_w[:, None]*affine, root_w*means, rcond=None)[0]
    y_scale = max(1., float(np.max(np.abs(means-trend))))
    response = (means-trend)/y_scale
    d = slope_matrix(unit_x)
    if len(d):
        c = d/np.linalg.norm(d/root_w[None, :], axis=1)[:, None]
        a = c.T/root_w[:, None]
        condition = float(np.linalg.cond(a))
        if condition > 1e10:
            raise ValueError("weighted constraint matrix is too ill-conditioned (condition > 1e10)")
        q, solver_status, work = _active_nnls(a, -root_w*response, max_steps)
        correction = (a @ q)/root_w
    else:
        c, a = d, np.empty((len(grid), 0))
        condition = 1.0
        q, correction = np.empty(0), np.zeros(len(grid))
        solver_status = "stationary"
        work = {"least_squares_solves": 0, "additions": 0, "removals": 0}
    fitted_scaled = response+correction
    fitted = trend+y_scale*fitted_scaled
    slack = c @ fitted_scaled
    objective = .5*float(np.dot(w*correction, correction))
    dual = -.5*float(np.dot(a @ q, a @ q))-float(np.dot(q, c @ response))
    primal_violation = float(max(0., -float(np.min(slack)))) if len(slack) else 0.
    complementarity = float(np.max(np.abs(q*slack))) if len(slack) else 0.
    stationarity = float(np.max(np.abs(root_w*correction-a @ q)))
    gap = objective-dual
    kkt_ok = primal_violation <= 1e-8 and complementarity <= 1e-8*max(1., objective) and abs(gap) <= 1e-8*max(1., objective) and stationarity <= 1e-8
    status = "converged" if solver_status == "stationary" and kkt_ok else ("iteration_limit" if solver_status == "iteration_limit" else "numerical_failure")
    slopes = np.diff(fitted)/np.diff(grid)
    grouped_sse = float(np.dot(raw_weights*(fitted-means), fitted-means))
    return {"status": status, "x": grid.tolist(), "group_means": means.tolist(),
            "group_weights": raw_weights.tolist(), "fitted": fitted.tolist(), "slopes": slopes.tolist(),
            "slope_increases": np.diff(slopes).tolist(), "observation_count": grouped["observation_count"],
            "distinct_count": len(grid), "within_group_sse": grouped["within_group_sse"],
            "weighted_sse": grouped_sse+grouped["within_group_sse"],
            "scaled_multipliers": q.tolist(), "scaled_slack": slack.tolist(),
            "diagnostics": {"primal_violation": primal_violation, "stationarity_residual": stationarity,
                            "max_complementarity": complementarity, "primal_dual_gap": gap,
                            "primal_objective": objective, "dual_objective": dual,
                            "kkt_ok": bool(kkt_ok), "constraint_condition": condition,
                            "response_scale": y_scale, "weight_scale": w_scale,
                            "tolerance": 1e-8, **work},
            "scope": "Numerical convex projection; not a test of convexity or a confidence interval."}


def predict(model, x):
    """Piecewise-linear interpolation, strictly within the observed domain."""
    if not isinstance(model, dict) or model.get("status") != "converged":
        raise ValueError("prediction requires a converged fit")
    points = _vector(x, "prediction x", maximum=10000)
    grid = np.asarray(model["x"])
    if np.any(points < grid[0]) or np.any(points > grid[-1]):
        raise ValueError("extrapolation is not supported")
    return np.interp(points, grid, model["fitted"])
