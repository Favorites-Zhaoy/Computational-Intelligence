"""Independent numerical reproduction of the unchanged 64-rule Mamdani FIS.

Requires: numpy, pandas, scipy, scikit-learn, xlrd.
Run: python fuzzy_cv_repro.py [scoredata.xls]
The 101-point output grid approximates MATLAB evalfis default sampling; final
numbers must be checked in MATLAB before claiming exact MATLAB performance.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold

DEFAULT_DATA = Path(__file__).resolve().parent / "hw1" / "scoredata.xls"
DATA_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA

# ruleTable(f, l, a), with output labels 1..4.
RULE_TABLE = np.array(
    [
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 2]],
        [[1, 1, 1, 1], [1, 2, 2, 2], [2, 2, 3, 3], [2, 3, 3, 4]],
        [[1, 1, 1, 2], [2, 2, 2, 3], [2, 3, 3, 3], [3, 3, 4, 4]],
        [[1, 1, 2, 2], [2, 2, 2, 3], [2, 3, 3, 4], [3, 4, 4, 4]],
    ],
    dtype=int,
)
# Python order is (attendance label, lab label, final label, output label).
RULES = np.array(
    [
        (a, l, f, RULE_TABLE[a, f, l] - 1)
        for a in range(4)
        for l in range(4)
        for f in range(4)
    ],
    dtype=int,
)
assert RULES.shape == (64, 4)
INPUT_CENTERS = np.array([55.0, 70.0, 82.0, 93.0])
OUTPUT_GRID = np.linspace(0.0, 100.0, 101)
OUTPUT_SIGMA = 4.0

def predict(parameters, features):
    """Mamdani min AND / min implication / max aggregation / centroid."""
    input_sigma = parameters[:3]
    output_centers = parameters[3:7]
    membership = np.exp(
        -0.5
        * (
            (features[:, :, None] - INPUT_CENTERS[None, None, :])
            / input_sigma[None, :, None]
        ) ** 2
    )
    activation = np.minimum(
        np.minimum(
            membership[:, 0, RULES[:, 0]], membership[:, 1, RULES[:, 1]]
        ),
        membership[:, 2, RULES[:, 2]],
    )
    by_output = np.stack(
        [
            activation[:, RULES[:, 3] == label].max(axis=1)
            for label in range(4)
        ],
        axis=1,
    )
    output_membership = np.exp(
        -0.5
        * ((OUTPUT_GRID[None, :] - output_centers[:, None]) / OUTPUT_SIGMA) ** 2
    )
    aggregated = np.max(
        np.minimum(by_output[:, :, None], output_membership[None, :, :]),
        axis=1,
    )
    return (aggregated @ OUTPUT_GRID) / aggregated.sum(axis=1)

def metrics(prediction, truth):
    error = prediction - truth
    return float(np.sqrt(np.mean(error**2))), float(np.mean(np.abs(error)))

data = pd.read_excel(DATA_FILE).to_numpy(dtype=float)
features = data[:, 1:4]
truth = data[:, 4]
assert features.shape == (68, 3)
assert np.max(np.abs(truth - features @ [0.16, 0.24, 0.60])) < 1e-10

baseline = np.array([7.0, 7.0, 7.0, 55.0, 70.0, 82.0, 93.0])
lower = np.array([4.0, 4.0, 4.0, 45.0, 62.0, 76.0, 88.0])
upper = np.array([16.0, 16.0, 16.0, 63.0, 78.0, 88.0, 99.0])

def fit(train_features, train_truth):
    return least_squares(
        lambda parameters: predict(parameters, train_features) - train_truth,
        baseline,
        bounds=(lower, upper),
        max_nfev=150,
        ftol=1e-7,
    ).x

# An identical input tuple must never appear on both sides of a fold.
_, groups = np.unique(features, axis=0, return_inverse=True)
splitter = GroupKFold(n_splits=5)
baseline_oof = np.empty(len(truth))
optimized_oof = np.empty(len(truth))
print(f"Rows: {len(truth)}, distinct input tuples: {len(np.unique(groups))}")
print("Baseline, all rows RMSE/MAE:", metrics(predict(baseline, features), truth))
for fold, (train, test) in enumerate(splitter.split(features, truth, groups), 1):
    fitted = fit(features[train], truth[train])
    baseline_oof[test] = predict(baseline, features[test])
    optimized_oof[test] = predict(fitted, features[test])
    print(
        f"Fold {fold}: test={len(test)}, "
        f"baseline RMSE/MAE={metrics(baseline_oof[test], truth[test])}, "
        f"optimized RMSE/MAE={metrics(optimized_oof[test], truth[test])}, "
        f"parameters={np.round(fitted, 4)}"
    )
print("Pooled out-of-fold baseline RMSE/MAE:", metrics(baseline_oof, truth))
print("Pooled out-of-fold optimized RMSE/MAE:", metrics(optimized_oof, truth))
full_fit = fit(features, truth)
print("Full-data fit parameters:", full_fit)
print("Full-data fit RMSE/MAE:", metrics(predict(full_fit, features), truth))
