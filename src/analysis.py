"""mAb production surrogate modelling, cross-validation, and profit optimization.

Expects exactly 30 selected batch_####.csv files plus initial_conditions.csv
inside datasets/. Exports reusable evidence to tables/ and figures/.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42
INITIAL = ["Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]"]
INPUTS = INITIAL + ["Time [d]"]
TARGET = "mAb [mg/L]"
DOWNTIME_DAYS = 1.0


def culture_cost(glucose, glutamine, vcd):
    """USD per litre, as specified by the assignment."""
    return 10 * np.asarray(glucose) + 20 * np.asarray(glutamine) + 0.0005 * np.asarray(vcd)


def load_data(project: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = project / "datasets"
    conditions = pd.read_csv(folder / "initial_conditions.csv")
    required = {"Batch ID", *INITIAL}
    if not required.issubset(conditions):
        raise ValueError(f"initial_conditions.csv lacks: {sorted(required - set(conditions))}")
    indexed = conditions.set_index("Batch ID", verify_integrity=True)
    files = sorted(folder.glob("batch_*.csv"))
    if len(files) != 30:
        raise ValueError(f"Expected exactly 30 batch files; found {len(files)} in {folder}")
    dfs = []
    for path in files:
        bid = int(path.stem.split("_")[1])
        if bid not in indexed.index:
            raise ValueError(f"Batch {bid} missing from initial_conditions.csv")
        df = pd.read_csv(path)
        if not {"Time [h]", TARGET}.issubset(df.columns):
            raise ValueError(f"Missing time/mAb column in {path.name}")
        df = df.sort_values("Time [h]").copy()
        df["Batch ID"] = bid
        df["Time [d]"] = df["Time [h]"] / 24.0
        for col in INITIAL:
            df[col] = indexed.at[bid, col]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    if data[INPUTS + [TARGET]].isna().any().any():
        raise ValueError("Missing data found in the model inputs or mAb measurements")
    if data["Time [d]"].min() < 0:
        raise ValueError("Negative harvest time is invalid")
    return data, conditions


def candidate_models():
    # Feature standardization takes place inside CV training folds.
    # Two different architectures and two polynomial complexities are compared.
    return {
        "Ridge polynomial (degree 2)": make_pipeline(
            StandardScaler(), PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=10.0)),
        "Ridge polynomial (degree 3)": make_pipeline(
            StandardScaler(), PolynomialFeatures(degree=3, include_bias=False),
            Ridge(alpha=30.0)),
        "Random forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, max_features=1.0,
            random_state=SEED, n_jobs=-1),
    }


def compare_models(data, table_dir):
    # Model time-averaged production rate, then reconstruct accumulated mAb.
    # This enforces zero predicted mAb at t=0, consistent with initial conditions.
    fit_rows = data["Time [d]"].to_numpy(dtype=float) > 0
    fit_data = data.loc[fit_rows].copy()
    x = fit_data[INPUTS]
    y = fit_data[TARGET].to_numpy(dtype=float)
    t = fit_data["Time [d]"].to_numpy(dtype=float)
    y_rate = y / t
    groups = fit_data["Batch ID"].to_numpy()
    folds = GroupKFold(n_splits=5)
    results = []
    predictions = {}
    for name, model in candidate_models().items():
        oof = np.full(len(fit_data), np.nan)
        for train_idx, test_idx in folds.split(x, y, groups):
            fitted = clone(model)
            fitted.fit(x.iloc[train_idx], y_rate[train_idx])
            oof[test_idx] = np.maximum(0, fitted.predict(x.iloc[test_idx]) * t[test_idx])
        rmse = float(np.sqrt(mean_squared_error(y, oof)))
        r2 = float(r2_score(y, oof))
        results.append({"Model": name, "Group CV folds": 5,
                        "OOF RMSE [mg/L]": rmse, "OOF R2": r2})
        predictions[name] = oof
        print(f"  {name}: RMSE={rmse:.3f} mg/L, R²={r2:.3f}")
    result_df = pd.DataFrame(results).sort_values("OOF RMSE [mg/L]").reset_index(drop=True)
    result_df.to_csv(table_dir / "model_comparison.csv", index=False)
    best_name = result_df.iloc[0]["Model"]
    best_model = clone(candidate_models()[best_name]).fit(x, y_rate)
    best_oof = predictions[best_name]
    pd.DataFrame({"Batch ID": groups, "Time [d]": fit_data["Time [d]"],
                  "Observed mAb [mg/L]": y, "Group CV predicted mAb [mg/L]": best_oof,
                  "Residual [mg/L]": y - best_oof}).to_csv(
        table_dir / "held_out_predictions.csv", index=False)
    # Empirical 2-sided, pointwise PIs using errors observed on entirely held-out batches.
    # These are approximate PIs; correlation within each batch and model selection
    # mean they do not have a formal guarantee of 95% coverage.
    residuals = y - best_oof
    low_residual, high_residual = np.quantile(residuals, [0.025, 0.975])
    return best_model, best_name, float(low_residual), float(high_residual), result_df


def optimizer_grid(conditions, observed_times):
    # Only choose initial conditions present in the provided experimental grid.
    levels = [np.sort(conditions[col].unique()) for col in INITIAL]
    combos = np.array(list(product(*levels)), dtype=float)
    times = np.sort(np.unique(observed_times))
    times = times[times > 0]  # Harvesting at t=0 cannot produce useful mAb.
    n_t = len(times)
    grid = pd.DataFrame({
        INITIAL[0]: np.repeat(combos[:, 0], n_t),
        INITIAL[1]: np.repeat(combos[:, 1], n_t),
        INITIAL[2]: np.repeat(combos[:, 2], n_t),
        "Time [d]": np.tile(times, len(combos)),
    })
    return grid


def predict_economics(frame, model, residual_low, residual_high):
    out = frame.copy()
    # Estimated cumulative mAb = elapsed days × estimated average production rate.
    mabs = np.maximum(0, np.asarray(model.predict(out[INPUTS]), dtype=float)
                      * out["Time [d]"].to_numpy(dtype=float))
    out["Predicted mAb [mg/L]"] = mabs
    out["mAb PI lower [mg/L]"] = np.maximum(0, mabs + residual_low)
    out["mAb PI upper [mg/L]"] = np.maximum(0, mabs + residual_high)
    cost = culture_cost(out[INITIAL[0]], out[INITIAL[1]], out[INITIAL[2]])
    divisor = out["Time [d]"].to_numpy(dtype=float) + DOWNTIME_DAYS
    out["Cell culture cost [$/L]"] = cost
    for label, mab in [("", mabs),
                       (" PI lower", out["mAb PI lower [mg/L]"].to_numpy()),
                       (" PI upper", out["mAb PI upper [mg/L]"].to_numpy())]:
        out[f"Cell culture value{label} [$/L]"] = 1000 * mab
        out[f"Space-time profit{label} [$/L/d]"] = (1000 * mab - cost) / divisor
        out[f"Space-time yield{label} [mg/L/d]"] = mab / divisor
    return out


def export_figures(data, scored, best, model_name, fig_dir):
    # True observed trajectories, labelled by batch ID.
    fig, ax = plt.subplots(figsize=(9, 5))
    for bid, sub in data.groupby("Batch ID"):
        ax.plot(sub["Time [d]"], sub[TARGET], alpha=0.35, linewidth=1)
    ax.set(xlabel="Time [days]", ylabel="Observed mAb [mg/L]",
           title="Observed mAb production: 30 independent batches")
    fig.tight_layout()
    fig.savefig(fig_dir / "observed_trajectories.png", dpi=170)
    plt.close(fig)

    # Profit versus harvest time at winning initial conditions.
    mask = np.ones(len(scored), dtype=bool)
    for col in INITIAL:
        mask &= scored[col].to_numpy() == best[col]
    trajectory = scored.loc[mask].sort_values("Time [d]")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = trajectory["Time [d]"].to_numpy()
    ax.plot(x, trajectory["Space-time profit [$/L/d]"], label="Predicted profit")
    ax.fill_between(x, trajectory["Space-time profit PI lower [$/L/d]"].to_numpy(),
                    trajectory["Space-time profit PI upper [$/L/d]"].to_numpy(),
                    alpha=0.20, label="Approximate 95% prediction interval")
    ax.axvline(float(best["Time [d]"]), color="black", linestyle="--", label="Recommended harvest")
    ax.set(xlabel="Harvest time [days]", ylabel="Space-time profit [$/L/day]",
           title=f"Recommended initial conditions ({model_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "optimal_harvest_time.png", dpi=170)
    plt.close(fig)

    # Heatmap of optimal profit for each glucose/glutamine pair, at recommended VCD.
    subset = scored[scored[INITIAL[2]] == best[INITIAL[2]]]
    per_pair = (subset.groupby(INITIAL[:2])["Space-time profit [$/L/d]"]
                .max().unstack(INITIAL[1]).sort_index().sort_index(axis=1))
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(per_pair.values, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(per_pair.columns)), labels=[f"{v:g}" for v in per_pair.columns])
    ax.set_yticks(np.arange(len(per_pair.index)), labels=[f"{v:g}" for v in per_pair.index])
    ax.set(xlabel="Initial glutamine [mM]", ylabel="Initial glucose [mM]",
           title="Maximum predicted space-time profit at recommended initial VCD")
    fig.colorbar(im, ax=ax, label="$/L/day")
    fig.tight_layout()
    fig.savefig(fig_dir / "profit_heatmap.png", dpi=170)
    plt.close(fig)


def run(project: Path):
    table_dir = project / "tables"
    fig_dir = project / "figures"
    table_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    (table_dir / "environment_versions.txt").write_text(
        "\n".join([
            f"Python {sys.version.split()[0]}",
            f"matplotlib {matplotlib.__version__}",
            f"numpy {np.__version__}",
            f"pandas {pd.__version__}",
            f"scikit-learn {sklearn.__version__}",
        ]) + "\n", encoding="utf8")
    data, conditions = load_data(project)
    print(f"Loaded {data['Batch ID'].nunique()} batches, {len(data)} time measurements")
    print("Comparing models with 5-fold batch-grouped cross-validation...")
    model, name, residual_low, residual_high, scores = compare_models(data, table_dir)
    print(f"Selected model: {name}")
    print("Searching initial-condition grid and all observed harvest times...")
    grid = optimizer_grid(conditions, data["Time [d]"].to_numpy())
    scored = predict_economics(grid, model, residual_low, residual_high)
    ranked = scored.sort_values("Space-time profit [$/L/d]", ascending=False)
    best = ranked.iloc[0]
    ranked.head(20).to_csv(table_dir / "top_20_options.csv", index=False)
    pd.DataFrame([best]).to_csv(table_dir / "recommended_conditions.csv", index=False)

    # Best harvest time for each of the 343 combinations.
    best_per_combo = (ranked.drop_duplicates(subset=INITIAL, keep="first")
                      .sort_values(INITIAL).reset_index(drop=True))
    best_per_combo.to_csv(table_dir / "profit_by_initial_conditions.csv", index=False)
    export_figures(data, scored, best, name, fig_dir)

    labels = [
        (INITIAL[0], "mM"), (INITIAL[1], "mM"), (INITIAL[2], "cells/mL"),
        ("Time [d]", "days"), ("Cell culture cost [$/L]", "$/L"),
        ("Cell culture value [$/L]", "$/L"),
        ("Cell culture value PI lower [$/L]", "$/L"),
        ("Cell culture value PI upper [$/L]", "$/L"),
        ("Space-time profit [$/L/d]", "$/L/day"),
        ("Space-time profit PI lower [$/L/d]", "$/L/day"),
        ("Space-time profit PI upper [$/L/d]", "$/L/day"),
        ("Space-time yield [mg/L/d]", "mg/L/day"),
        ("Space-time yield PI lower [mg/L/d]", "mg/L/day"),
        ("Space-time yield PI upper [mg/L/d]", "mg/L/day"),
    ]
    lines = ["# Model and optimization results", "",
             f"Selected model: **{name}**.", "",
             "Validation: 5-fold grouped cross-validation; each held-out fold contains entire batches.",
             "Models predict mAb/time to enforce zero mAb at t=0; fit excludes noisy t=0 measurements.",
             "Prediction intervals: approximate pointwise 95% intervals based on the 2.5th and 97.5th",
             "percentiles of held-out residuals, added to the final model prediction. These intervals",
             "are not formal guarantees of coverage, especially for unobserved initial-condition combinations.",
             "", "## Recommended conditions", ""]
    for col, unit in labels:
        lines.append(f"- {col}: **{float(best[col]):,.3f} {unit}**")
    lines += ["", "## Model comparison", ""]
    for _, row in scores.iterrows():
        lines.append(f"- {row['Model']}: RMSE {row['OOF RMSE [mg/L]']:.3f} mg/L, R² {row['OOF R2']:.3f}")
    lines += ["", "See figures/ and tables/ for the underlying evidence.", ""]
    summary_path = table_dir / "results_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf8")

    print("\n=== RECOMMENDED CONDITIONS (MODEL ESTIMATES) ===")
    for col, unit in labels:
        print(f"{col}: {float(best[col]):,.3f} {unit}")
    print("\nPrediction intervals are approximate, not guaranteed coverage.")
    print(f"Saved results to {table_dir}/ and figures to {fig_dir}/")
    return best
