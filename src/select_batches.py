"""Select and copy 30 mAb production batches using a reproducible design.

Run from the project's Terminal:
    python src/select_batches.py /path/to/provided_files

If no path is supplied, the script asks you to paste or drag the folder path.
"""

from itertools import product
from pathlib import Path
import argparse
import shutil
import shlex

import numpy as np
import pandas as pd

RANDOM_SEED = 42
FEATURES = ["Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]"]


def choose_design(conditions, rng):
    # Each initial-condition combination has three simulated replicates.
    # Work with unique combinations first, so every selected batch tests a
    # different glucose/glutamine/VCD setting.
    levels = [sorted(conditions[col].unique()) for col in FEATURES]
    if any(len(v) != 7 for v in levels):
        raise ValueError("Expected exactly 7 levels for each initial condition.")

    available = set(map(tuple, conditions[FEATURES].to_numpy()))
    selected = []
    selected_set = set()

    def add(point, group):
        point = tuple(point)
        if point not in available:
            raise ValueError(f"Initial condition is missing: {point}")
        if point in selected_set:
            raise ValueError(f"Duplicate initial condition: {point}")
        selected.append((point, group))
        selected_set.add(point)

    # Eight full-range corners.
    for point in product(*[(v[0], v[-1]) for v in levels]):
        add(point, "Corner")

    # The two face centres for each of the three experimental axes.
    middle = tuple(v[3] for v in levels)
    for axis in range(3):
        for boundary in (levels[axis][0], levels[axis][-1]):
            point = list(middle)
            point[axis] = boundary
            add(point, "Face centre")

    add(middle, "Overall centre")

    # Greedy maximin sampling picks 10 spread-out points strictly inside the
    # 3-D parameter ranges. Distances are normalized so VCD doesn't dominate.
    interior = list(product(*[v[1:-1] for v in levels]))
    ranges = np.array([v[-1] - v[0] for v in levels], dtype=float)
    mins = np.array([v[0] for v in levels], dtype=float)

    def scaled(points):
        return (np.asarray(points, dtype=float) - mins) / ranges

    for _ in range(10):
        candidates = [p for p in interior if p not in selected_set]
        chosen = scaled([p for p, _ in selected])
        candidate_array = scaled(candidates)
        nearest_distances = np.linalg.norm(
            candidate_array[:, None, :] - chosen[None, :, :], axis=2
        ).min(axis=1)
        # Random tie-breaking, but reproducible due to the fixed seed.
        best = np.flatnonzero(np.isclose(nearest_distances, nearest_distances.max()))
        add(candidates[int(rng.choice(best))], "Distributed interior")

    # Last five combinations are genuinely random among ALL remaining grid
    # locations (interior, face, edge, or corner), sampled without replacement.
    remaining = sorted(available - selected_set)
    random_indices = rng.choice(len(remaining), size=5, replace=False)
    for index in random_indices:
        add(remaining[int(index)], "Random")

    if len(selected) != 30 or len(selected_set) != 30:
        raise AssertionError("Expected 30 unique initial-condition combinations.")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", help="Path to professor's provided_files folder")
    args = parser.parse_args()

    if args.source:
        source = Path(args.source).expanduser()
    else:
        raw = input("Drag the professor's provided_files folder here, then press Enter:\n> ").strip()
        # Dragging a macOS folder into Terminal often escapes spaces with backslashes.
        parts = shlex.split(raw)
        if len(parts) != 1:
            raise ValueError("Please supply exactly one folder path.")
        source = Path(parts[0]).expanduser()

    if not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")

    project = Path(__file__).resolve().parent.parent
    destination = project / "datasets"
    table_folder = project / "tables"
    conditions_file = destination / "initial_conditions.csv"
    if not conditions_file.is_file():
        raise FileNotFoundError(f"Missing initial conditions: {conditions_file}")

    conditions = pd.read_csv(conditions_file)
    missing_columns = set(["Batch ID", *FEATURES]) - set(conditions.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in initial_conditions.csv: {missing_columns}")

    rng = np.random.default_rng(RANDOM_SEED)
    design = choose_design(conditions, rng)
    records = []

    # Choose one of the three replicates randomly for each unique condition.
    for combo, group in design:
        matches = conditions.loc[(conditions[FEATURES] == list(combo)).all(axis=1)]
        if len(matches) != 3:
            raise ValueError(f"Expected 3 replicates for {combo}, found {len(matches)}")
        batch_id = int(rng.choice(matches["Batch ID"].to_numpy()))
        filename = f"batch_{batch_id:04d}.csv"
        records.append({
            "Selection type": group,
            "Batch ID": batch_id,
            FEATURES[0]: combo[0],
            FEATURES[1]: combo[1],
            FEATURES[2]: combo[2],
            "File": filename,
        })

    # Verify that every required file is available before copying anything.
    missing = [r["File"] for r in records if not (source / r["File"]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Can't find {len(missing)} batch file(s) in {source}. "
            f"First missing file: {missing[0]}"
        )

    destination.mkdir(exist_ok=True)
    table_folder.mkdir(exist_ok=True)
    existing = list(destination.glob("batch_*.csv"))
    if existing:
        raise FileExistsError(
            f"There are already {len(existing)} batch files in {destination}. "
            "Please move them out before running this selector to avoid extra batches."
        )

    for record in records:
        shutil.copy2(source / record["File"], destination / record["File"])

    selection_path = table_folder / "selected_batches.csv"
    pd.DataFrame(records).to_csv(selection_path, index=False)
    print("\nSuccess! Copied exactly 30 batch CSV files into datasets/.")
    print("Saved the selection details to tables/selected_batches.csv.")
    print("\nSelection summary:")
    print(pd.DataFrame(records).groupby("Selection type", sort=False).size().to_string())
    print("\nSelected batch IDs:")
    print(", ".join(str(r["Batch ID"]) for r in records))


if __name__ == "__main__":
    main()
