"""Deterministic synthetic experiments; no real dataset or inferential claims."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from convex_shape import fit, predict
from convex_shape.reference import exact_projection


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    x = np.array([-3, -2.7, -2.1, -1.95, -1.5, -1.1, -.7, -.1, 0, .05, .2, .8, 1.5, 1.7, 2, 2.2, 2.8, 3.4, 3.7, 4])
    truth = lambda a: .12*(a-1)**2+.35*a
    rng = np.random.default_rng(20261109)
    sigma = .25+.12*(x+3)/7
    observed_x = np.r_[x, .2, .2]
    observed_sigma = np.r_[sigma, .1, .6]
    observed_y = truth(observed_x)+rng.normal(size=len(observed_x))*observed_sigma
    weights = 1/observed_sigma**2
    correct = fit(observed_x, observed_y, weights)
    index_fit = fit(np.arange(len(x)), correct["group_means"], correct["group_weights"])
    wrong_grid_slopes = np.diff(index_fit["fitted"])/np.diff(x)
    query = np.linspace(x[0], x[-1], 401)
    predicted = predict(correct, query)
    wrong_predicted = np.interp(query, x, index_fit["fitted"])
    rmse = lambda a, b: float(np.sqrt(np.mean((np.array(a)-b)**2)))

    small_x, small_y = [0, 1, 4], [0, 2, 5]
    small_fit = fit(small_x, small_y)
    exact = exact_projection(small_x, small_y)
    duplicate = fit([0, 1, 1, 2, 3], [0, 8, 0, 2, 6], [1, 1, 9, 1, 1])
    # No observation noise: a wrong shape assumption alone causes this error.
    miss_x = np.arange(-3, 4)
    miss_y = -miss_x**2
    miss_fit = fit(miss_x, miss_y)
    budget_fit = fit(list(range(8)), [0, 4, 0, 4, 0, 4, 0, 4], max_steps=1)
    if any(m["status"] != "converged" for m in [correct, index_fit, small_fit, duplicate, miss_fit]):
        raise RuntimeError("A demonstration fit failed its numerical convergence checks")
    summary = {
        "data": "Deterministic synthetic design with seeded Gaussian perturbations; seed 20261109.",
        "truth": "f(x)=0.12*(x-1)^2+0.35*x", "weights": "inverse of the specified synthetic noise variances",
        "synthetic": {"observed_x": observed_x.tolist(), "observed_y": observed_y.tolist(),
                      "noise_sd": observed_sigma.tolist(), "fit": correct,
                      "index_based_fit": index_fit,
                      "index_fit_actual_grid_slopes": wrong_grid_slopes.tolist(),
                      "index_fit_actual_grid_convexity_violations": int(np.sum(np.diff(wrong_grid_slopes) < -1e-7)),
                      "convex_fit_grid_rmse": rmse(predicted, truth(query)),
                      "index_fit_grid_rmse": rmse(wrong_predicted, truth(query)),
                      "evaluation_grid_count": len(query)},
        "irregular_counterexample": {"x": small_x, "y": small_y, "fit": small_fit,
                                      "exact_fitted": [str(v) for v in exact["fitted"]],
                                      "exact_weighted_sse": str(exact["weighted_sse"]),
                                      "faces_examined": exact["faces_examined"]},
        "duplicate_example": duplicate,
        "noiseless_misspecification": {"x": miss_x.tolist(), "truth": miss_y.tolist(), "fit": miss_fit,
                                        "design_point_rmse": rmse(miss_fit["fitted"], miss_y)},
        "budget_exit_example": budget_fit,
        "scope": "RMSEs are descriptive synthetic examples, not confidence statements or general performance guarantees."
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (output/'predictions.json').write_text(json.dumps({"x": query.tolist(), "truth": truth(query).tolist(), "convex_fit": predicted.tolist(), "index_based_fit": wrong_predicted.tolist()}, indent=2, allow_nan=False)+'\n')

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    fig.suptitle("Convex regression: respect the design, question the shape", fontsize=17, fontweight="bold")
    blue, orange, purple = '#176885', '#b45822', '#7660a4'
    ax = axes[0, 0]
    ax.scatter(observed_x, observed_y, s=18, color='#999999', alpha=.8, label='Synthetic observations')
    ax.plot(query, truth(query), '--', color='#222222', label='Known synthetic truth')
    ax.plot(query, predicted, color=blue, lw=2, label='Weighted convex fit')
    ax.set(title='A. Convex fit on an irregular design', xlabel='x', ylabel='Response')
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    centers = (x[:-1]+x[1:])/2
    ax.plot(centers, wrong_grid_slopes, marker='.', color=orange, label='Fit treating indices as x')
    ax.plot(centers, correct['slopes'], marker='.', color=blue, label='Fit using actual x')
    ax.set(title='B. Adjacent slopes in actual x units', xlabel='Interval midpoint', ylabel='Slope')
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    ax.plot(small_x, small_y, 'o--', color=orange, label='Index-convex values: [0, 2, 5]')
    ax.plot(small_x, small_fit['fitted'], 'o-', color=blue, label='Correct convex projection')
    ax.set(title='C. Positive second differences can mislead', xlabel='x = [0, 1, 4]', ylabel='Response')
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    ax.plot(miss_x, miss_y, 'o--', color=purple, label='Noiseless concave truth')
    ax.plot(miss_x, miss_fit['fitted'], 'o-', color=blue, label='Convex fit: constant -4')
    ax.set(title='D. A wrong shape assumption causes error', xlabel='x', ylabel='Response')
    ax.legend(fontsize=9)
    fig.savefig(output/'shape_audit.png', dpi=160)
    plt.close(fig)
    print(json.dumps({"fit_status": correct['status'], "least_squares_solves": correct['diagnostics']['least_squares_solves'],
                      "convex_fit_grid_rmse": summary['synthetic']['convex_fit_grid_rmse'],
                      "index_fit_grid_rmse": summary['synthetic']['index_fit_grid_rmse'],
                      "index_fit_actual_grid_convexity_violations": summary['synthetic']['index_fit_actual_grid_convexity_violations'],
                      "noiseless_misspecification_rmse": summary['noiseless_misspecification']['design_point_rmse'],
                      "budget_exit_status": budget_fit['status']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results'))
    run(parser.parse_args().output)
