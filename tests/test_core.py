from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import nnls

from convex_shape import aggregate, fit, predict, slope_matrix
from convex_shape.reference import exact_projection


def scipy_projection(x, y, weights):
    # Independently form raw slope-difference constraints, without normalization
    # or affine detrending from the production path.
    x, y, weights = map(np.asarray, (x, y, weights))
    n = len(x)
    d = np.zeros((n-2, n))
    for j in range(n-2):
        d[j, j] = 1/(x[j+1]-x[j])
        d[j, j+2] = 1/(x[j+2]-x[j+1])
        d[j, j+1] = -d[j, j]-d[j, j+2]
    multipliers = nnls(d.T/np.sqrt(weights)[:, None], -y*np.sqrt(weights), maxiter=10000)[0]
    return y+(d.T @ multipliers)/weights


def test_irregular_counterexample_exact():
    # Index-based second differences say convex; actual adjacent slopes do not.
    x, y = [0, 1, 4], [0, 2, 5]
    assert np.diff(y, n=2)[0] > 0
    assert (slope_matrix(x) @ y)[0] < 0
    out = fit(x, y)
    reference = exact_projection(x, y)
    assert reference["fitted"] == [Fraction(9, 26), Fraction(20, 13), Fraction(133, 26)]
    assert reference["weighted_sse"] == Fraction(9, 26)
    np.testing.assert_allclose(out["fitted"], list(map(float, reference["fitted"])), atol=1e-12)
    assert out["status"] == "converged"


@pytest.mark.parametrize("seed", range(10))
def test_small_fits_against_exact_face_enumeration(seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.integers(1, 5, 6)).tolist()
    y = rng.integers(-8, 9, 6).tolist()
    w = rng.integers(1, 5, 6).tolist()
    actual = fit(x, y, w)
    reference = exact_projection(x, y, w)
    assert reference["faces_examined"] == 16
    assert actual["status"] == "converged"
    np.testing.assert_allclose(actual["fitted"], list(map(float, reference["fitted"])), atol=2e-10)
    assert actual["weighted_sse"] == pytest.approx(float(reference["weighted_sse"]), abs=1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_larger_fits_against_scipy(seed):
    rng = np.random.default_rng(seed+20)
    x = np.cumsum(rng.uniform(.2, 2, 25))
    y = np.sin(x)+rng.normal(size=25)
    w = rng.uniform(.5, 4, 25)
    actual = fit(x, y, w)
    expected = scipy_projection(x, y, w)
    assert actual["status"] == "converged"
    np.testing.assert_allclose(actual["fitted"], expected, atol=2e-8)
    assert min(actual["slope_increases"]) >= -2e-8


def test_exactly_convex_data_unchanged():
    x = np.array([-3, -1, 0, .1, 2, 5])
    y = x*x+2*x-4
    out = fit(x, y)
    np.testing.assert_allclose(out["fitted"], y, atol=1e-12)
    assert out["diagnostics"]["least_squares_solves"] == 0


def test_affine_data_and_two_points():
    for x in ([0, 7], [-3, 0, 1, 1.5, 8]):
        y = 5-2*np.array(x)
        out = fit(x, y)
        assert out["status"] == "converged"
        np.testing.assert_allclose(out["fitted"], y, atol=1e-12)


def test_weighted_residuals_orthogonal_to_affine_functions():
    x = np.array([0, 1, 2, 4, 5, 9])
    y = np.array([4, 1, 8, -2, 2, 0])
    w = np.array([1, 2, 3, 5, 1, 7])
    residual = w*(np.array(fit(x, y, w)["fitted"])-y)
    assert abs(residual.sum()) < 1e-10
    assert abs(residual @ x) < 1e-9


def test_affine_design_and_response_equivariance():
    x = np.array([0, 1, 2, 4, 7])
    y = np.array([0, 3, 1, 4, 2])
    w = [1, 4, 2, 1, 3]
    base = fit(x, y, w)
    changed = fit(x*100+20, y*3+5*x-10, w)
    np.testing.assert_allclose(changed["fitted"], 3*np.array(base["fitted"])+5*x-10, atol=1e-10)


def test_weight_rescaling_keeps_fit_and_scales_sse():
    x, y, w = [0, 1, 3, 5], [1, 8, 0, 2], np.array([1, 2, 5, 3])
    a, b = fit(x, y, w), fit(x, y, w*17)
    np.testing.assert_allclose(a["fitted"], b["fitted"], atol=1e-12)
    assert b["weighted_sse"] == pytest.approx(a["weighted_sse"]*17)


def test_duplicate_aggregation_retains_weighted_sse():
    x, y, w = [0, 1, 1, 2, 3], [0, 8, 0, 2, 6], [1, 1, 9, 1, 1]
    groups = aggregate(x, y, w)
    np.testing.assert_allclose(groups["y"], [0, .8, 2, 6])
    assert groups["within_group_sse"] == pytest.approx(57.6)
    out = fit(x, y, w)
    direct_sse = np.sum(np.array(w)*(predict(out, x)-y)**2)
    assert out["weighted_sse"] == pytest.approx(direct_sse)
    assert out["observation_count"] == 5 and out["distinct_count"] == 4


def test_permutation_invariance():
    x, y, w = [3, 1, 0, 1, 2], [6, 8, 0, 0, 2], [1, 1, 1, 9, 1]
    a, b = fit(x, y, w), fit(x[::-1], y[::-1], w[::-1])
    np.testing.assert_allclose(a["fitted"], b["fitted"], atol=1e-12)


def test_piecewise_linear_interpolation_and_no_extrapolation():
    out = fit([0, 1, 3], [0, 1, 9])
    np.testing.assert_allclose(predict(out, [0, .5, 1, 2, 3]), [0, .5, 1, 5, 9])
    for x in ([-.001], [3.001]):
        with pytest.raises(ValueError, match="extrapolation"):
            predict(out, x)


def test_budget_exit_never_predicts_as_converged():
    out = fit(list(range(8)), [0, 4, 0, 4, 0, 4, 0, 4], max_steps=1)
    assert out["status"] == "iteration_limit"
    assert out["diagnostics"]["least_squares_solves"] == 1
    with pytest.raises(ValueError, match="converged"):
        predict(out, [1])


def test_noiseless_misspecification_is_not_cured_by_optimization():
    x = np.arange(-3, 4)
    y = -x*x
    out = fit(x, y)
    assert out["status"] == "converged"
    np.testing.assert_allclose(out["fitted"], np.repeat(-4., 7), atol=1e-10)
    assert out["weighted_sse"] == pytest.approx(84)


@pytest.mark.parametrize("x,y,w", [([], [], None), ([1, 1], [0, 1], None), ([0, 1], [1], None), ([0, 1], [1, True], None), ([0, 1], [1, float("nan")], None), ([0, 1], [0, 1], [1, 0]), ([0, 1], [0, 1], [1, -1]), ([0, 1], [0, 1], [1, 1e-8]), ([0, 1e-9, 1], [0, 1, 2], None), ([0, 1], [0, 1e7], None), (list(range(81)), [1]*81, None), ([[0], [1]], [0, 1], None)])
def test_input_validation(x, y, w):
    with pytest.raises(ValueError):
        fit(x, y, w)


@pytest.mark.parametrize("budget", [0, -1, 1.5, True, 10001])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        fit([0, 1, 2], [0, 2, 1], max_steps=budget)


def test_exact_reference_rejects_unbounded_enumeration():
    with pytest.raises(ValueError):
        exact_projection(range(10), range(10))


@pytest.mark.parametrize("weights", [[1e-300, 1e-300], [1, 1e-300]])
def test_underflow_scale_weights_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        fit([0, 1], [0, 1], weights)


def test_underflow_scale_design_rejected():
    with pytest.raises(ValueError, match="span"):
        fit([0, 1e-310], [0, 1])
    with pytest.raises(ValueError, match="span"):
        slope_matrix([0, 1e-310])
