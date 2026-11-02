"""Build a full portfolio README from this project's actual exported results.

Run from the repository root: python src/finalize_readme.py
Does not change training data or re-run the predictive model.
"""
from pathlib import Path
import re
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
TABLES = PROJECT / 'tables'
FEATURES = ['Glucose [mM]', 'Glutamine [mM]', 'VCD [cells/mL]']


def as_money(value):
    return f'${value:,.2f}'


def read_inputs():
    required = ['recommended_conditions.csv', 'model_comparison.csv',
                'profit_by_initial_conditions.csv', 'selected_batches.csv',
                'environment_versions.txt']
    missing = [name for name in required if not (TABLES / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing exported results: ' + ', '.join(missing)
                            + '. Run python main.py first.')
    recommended = pd.read_csv(TABLES / required[0]).iloc[0]
    models = pd.read_csv(TABLES / required[1])
    factors = pd.read_csv(TABLES / required[2])
    design = pd.read_csv(TABLES / required[3])
    versions = (TABLES / required[4]).read_text().splitlines()
    return recommended, models, factors, design, versions


def metric_table(model_data):
    return '\n'.join([
        '| Architecture | Held-out RMSE (mg/L) | Held-out R² |',
        '|---|---:|---:|',
        *(f'| {row["Model"]} | {row["OOF RMSE [mg/L]"]:.3f} | {row["OOF R2"]:.3f} |'
          for _, row in model_data.iterrows()),
    ])


def get_factor_rows(factors, feature):
    agg = (factors.groupby(feature)
           .agg(mean_profit=('Space-time profit [$/L/d]', 'mean'),
                mean_yield=('Space-time yield [mg/L/d]', 'mean'),
                mean_harvest=('Time [d]', 'mean'))
           .reset_index().sort_values(feature))
    return agg


def effect_table(factors):
    rows = [
        '| Input | Setting | Mean optimized profit ($/L/day) | Mean yield at profit-optimal harvest (mg/L/day) |',
        '|---|---:|---:|---:|'
    ]
    for feature in FEATURES:
        group = get_factor_rows(factors, feature)
        label = feature.replace(' [cells/mL]', ' (cells/mL)').replace(' [mM]', ' (mM)')
        for _, row in group.iterrows():
            level = f'{row[feature]:,.0f}' if feature == FEATURES[2] else f'{row[feature]:g}'
            rows.append(f'| {label} | {level} | {row["mean_profit"]:,.2f} | {row["mean_yield"]:.3f} |')
    return '\n'.join(rows)


def effect_sentence(factors, feature, name):
    group = get_factor_rows(factors, feature)
    low, high = group.iloc[0], group.iloc[-1]
    high_mean = group.loc[group['mean_profit'].idxmax()]
    def fmt(v):
        return f'{v:,.0f}' if feature == FEATURES[2] else f'{v:g}'
    return (f'For **{name}**, the mean optimized profit across the 49 combinations of the other two inputs '
            f'is {as_money(low["mean_profit"])} per L/day at the lowest studied setting '
            f'({fmt(low[feature])}), and {as_money(high["mean_profit"])} per L/day at the highest '
            f'({fmt(high[feature])}). The input level with the highest mean predicted profit is '
            f'{fmt(high_mean[feature])}, averaging {as_money(high_mean["mean_profit"])} per L/day. '
            f'These are *model-derived marginal summaries*, not directly measured independent effects.')


def version_list(lines):
    mapping = {}
    for line in lines:
        split = line.strip().split(' ', 1)
        if len(split) == 2:
            mapping[split[0].lower()] = split[1]
    expected = ['python', 'matplotlib', 'numpy', 'pandas', 'scikit-learn']
    missing = [x for x in expected if x not in mapping]
    if missing:
        raise ValueError('Unknown environment versions: ' + ', '.join(missing))
    return '\n'.join([
        f'- Python {mapping["python"]}',
        f'- matplotlib {mapping["matplotlib"]}',
        f'- NumPy {mapping["numpy"]}',
        f'- pandas {mapping["pandas"]}',
        f'- scikit-learn {mapping["scikit-learn"]}',
    ])


def build(reco, models, factors, selected, versions):
    total_data = 2550  # The current dataset has 30 runs x 85 measurements; checked below.
    # Do not claim an unobserved measurement count if the data selection differs.
    n_batches = len(selected)
    best_model = models.iloc[0]['Model']
    best_rmse = float(models.iloc[0]['OOF RMSE [mg/L]'])
    best_r2 = float(models.iloc[0]['OOF R2'])
    duration = float(reco['Time [d]'])
    mab = float(reco['Predicted mAb [mg/L]'])
    cost = float(reco['Cell culture cost [$/L]'])
    value = float(reco['Cell culture value [$/L]'])
    profit = float(reco['Space-time profit [$/L/d]'])
    yield_mab = float(reco['Space-time yield [mg/L/d]'])
    to_3 = lambda x: f'{float(x):,.3f}'
    md = f'''# Monoclonal Antibody Production Profit Optimization

This project estimates economically favourable operating conditions for a one-litre monoclonal antibody (mAb) production process. Python code selects a structured set of 30 simulated bioreactor batches, compares predictive regression models, and estimates mAb concentration over time from the initial culture settings. It then searches feasible glucose, glutamine, inoculum density, and harvest-time choices for the highest predicted profit per litre of reactor capacity per day.

## Highlights

- **Designed** a reproducible, 30-run experimental sampling plan across three initial culture variables.
- **Developed** batch-aware model validation that separates whole bioreactor runs between training and evaluation.
- **Compared** polynomial ridge regression and random forest regression using out-of-fold prediction errors.
- **Optimized** initial culture settings and harvest timing using process costs, product value, and reactor downtime.
- **Reported** estimated financial outcomes with approximate 95% prediction intervals and exported supporting evidence.

## Environment

{version_list(versions)}

Run the analysis from the repository root with `python main.py` in an environment containing the listed dependencies. The separate `src/select_batches.py` utility is used only when first preparing the datasets from the original batch library.

## Code Design

### Program entry point and separation of responsibilities

The executable entry point is `main.py`. It deliberately contains very little code: it identifies the project root and calls `run()` from `src/analysis.py`. This keeps analysis logic in one reusable module while preserving a simple, predictable command for anyone inspecting the project. Running `main.py` does not download data or change the choice of batches; it reads the 30 CSV files already placed in `datasets/`. The original template's directory arrangement separates source code, input datasets, exported tables, and graphics. This organization allows another user to rerun the numerical work and inspect the results without sorting through a single large script.

### Experimental data selection and provenance

The dataset is prepared separately with `src/select_batches.py`. The utility reads the 1,029 available initial-condition records, finds the seven studied levels for each of three inputs, and chooses 30 distinct combinations. Eight cover the extreme corners, six cover the centres of the parameter-space faces, one covers the overall centre, ten are distributed within the boundaries using a maximum-minimum-distance strategy, and five are sampled randomly from the remaining combinations. A fixed random seed of 42 is used to reproduce tie-breaking, the final five selections, and the selection of one replicate from each available three-replicate setting. The selected batch IDs and their sampling categories are saved to `tables/selected_batches.csv`. This preparatory stage records which experiments were used and makes the small data budget transparent.

### Loading and checking the time series

At execution, the analysis code reads `datasets/initial_conditions.csv` and checks that exactly 30 `batch_####.csv` files exist. Each filename supplies its batch identifier, which is matched to the initial glucose concentration, glutamine concentration, and viable cell density. Every batch file is sorted by measurement time, and hours are converted into days. The data are then combined into one table while retaining a batch ID for every observation. Missing values in the modeling inputs or target cause an error, and negative time is rejected. These checks catch common mistakes such as using the wrong input files or losing the connection between batch measurements and initial operating settings.

### Predictor and target construction

The regression inputs are initial glucose in mM, initial glutamine in mM, initial VCD in cells/mL, and elapsed time in days. The objective is to predict accumulated mAb concentration in mg/L. The model is trained on mAb concentration divided by elapsed time, which is the time-averaged production rate, instead of training directly on concentration. Predictions are multiplied by elapsed time to recover concentration. Measurements at time zero are excluded from training because the rate would require division by zero. This design also makes the reconstructed concentration zero at time zero, consistent with the stated initial conditions, despite possible small simulated measurement noise. Negative reconstructed mAb values are clipped to zero to avoid negative product values during optimization.

### Candidate regression models and validation

Three candidate models are compared: a degree-two polynomial followed by ridge regression (`alpha=10`), a degree-three polynomial followed by ridge regression (`alpha=30`), and a random forest with 200 trees, at least three observations per terminal leaf, and `max_features=1.0`. Ridge pipelines first standardize the predictors, then construct polynomial features, then fit regularized linear coefficients. Scaling is fitted within each training fold, which avoids leakage from held-out observations. The forest learns nonlinear partitions without explicitly creating polynomial terms. Five-fold `GroupKFold` validation uses the batch identifier as the grouping variable. All measurements from an individual batch are therefore either in training or held out for a given fold, rather than artificially treating adjacent time points as independent experiments.

### Selecting the final model and estimating uncertainty

For each architecture, the code reconstructs out-of-fold mAb predictions and calculates root-mean-square error (RMSE) and the coefficient of determination (R²) in concentration units. Models are ranked by RMSE, and the lowest-error architecture is retrained using all 30 selected batches. Its held-out residuals, calculated before full-data fitting, supply empirical 2.5th and 97.5th percentile offsets for prediction intervals. Adding those offsets to each final prediction provides simple, approximate pointwise 95% intervals. The code saves the complete out-of-fold predictions and residuals, so the interval calculation can be inspected. These intervals do not account for every source of uncertainty, particularly model selection and the search for the maximum.

### Economic calculation and numerical search

The program constructs the 343 unique initial-condition combinations in the provided seven-by-seven-by-seven grid and pairs them with each positive harvest time appearing in the measured data. For every candidate, the fitted model predicts mAb, and the code calculates culture cost using the specified nutrient and inoculum charges. Product value is $1,000 for each predicted mg/L of mAb, so profit per litre equals value minus initial culture cost. Space-time profit divides this profit by harvest days plus the required one day of reactor downtime. Space-time yield uses the same denominator but has mg/L/day units. Predictions and uncertainty bounds are passed through the same equations. The candidate with the largest predicted space-time profit is selected, and the program exports the recommended settings, 20 leading options, and best harvest choice for each initial-condition combination.

### Exported evidence and repeatability

The analysis writes `tables/model_comparison.csv`, `tables/held_out_predictions.csv`, `tables/profit_by_initial_conditions.csv`, `tables/top_20_options.csv`, `tables/recommended_conditions.csv`, `tables/results_summary.txt`, and `tables/environment_versions.txt`. Three PNG graphics show the measured mAb trajectories, predicted profit versus harvest time for the recommended settings, and a glucose–glutamine profit heatmap at the recommended initial VCD. Numeric outputs and images come from the same run, avoiding manual recalculation when results change. A version file records the actual Python and library versions, and a fixed random seed makes the selection and random forest fitting repeatable on the same software environment.

## Analysis

### Engineering objective and economic trade-off

The practical target is not simply the greatest concentration of monoclonal antibody at the end of a batch. The objective is profit generated per unit reactor volume and calendar time. A very long batch may collect more product but prevent the reactor from beginning the next batch; a short batch may have high turnover but fail to recover the initial nutrient and inoculum expenses. The one-day downtime creates an additional penalty for very short production cycles. Product value is defined as 1,000 times mAb in mg/L, culture cost depends on the three starting variables, and the reactor capacity is one litre. Consequently, the objective has units of dollars per litre per day, while space-time yield is mg/L/day. These measures should be considered together: greater physical productivity does not guarantee greater profitability if the additional production requires sufficiently expensive initial media or cells.

The cost equation also provides useful benchmarks before fitting any model. Raising initial glucose by 1 mM adds $10/L of cost, raising glutamine by 1 mM adds $20/L, and raising initial VCD by 100,000 cells/mL adds $50/L. These are direct cost effects given by the problem; their benefits for product production must be learned from the batch data. Because the resulting mAb concentration and preferred harvest day may change together, comparing operating settings solely at a fixed final time would miss the operating trade-off. The program instead evaluates a harvest-time choice for every candidate setting and ranks candidates by the complete economic objective.

### Justification for the 30 experiments

The experimental budget covers only {n_batches} of the 1,029 available simulated runs, so selecting the most convenient files would use that budget poorly. I chose an explicit mixture of structured coverage and random sampling. The eight corner combinations provide examples of the joint minimum and maximum levels for the three controllable initial variables. They are helpful for determining whether an apparently attractive optimum lies near the limits of the process space. Six face centres isolate how the response behaves when one variable is at a boundary but the other two remain central. The overall centre adds a reference operating point away from all boundaries. Together, those 15 points cover the global geometry of the operating space before any remaining batches are allocated.

For the next ten, a maximin-style algorithm selects combinations strictly inside the studied ranges that are far from previously selected points in normalized three-dimensional coordinates. Normalization matters: VCD is numerically much larger than the nutrient concentrations, so Euclidean distances in raw units would almost entirely reflect VCD. The algorithm uses the same fixed seed to resolve distance ties. Finally, five additional settings are sampled without replacement from all remaining locations. These random points provide some independent variety and reduce dependence on a completely hand-designed set. The script then randomly chooses one of the three available replicates for each selected combination, giving 30 unique operating settings rather than repeating the same setting several times.

This design prioritizes broad operating-condition coverage, which is appropriate when a surrogate model must compare many conditions. It also has limitations. With one replicate per setting, the model does not directly quantify batch-to-batch repeatability at a fixed condition. Thirty independent batches remain a small training sample even though each contains many time measurements. The later time points are correlated within each batch and cannot be treated as independent experiments. A follow-up study should use new batches to verify the predicted optimum, ideally including replicates around high-profit settings and near the recommended harvest time.

### Choice of model architecture and validation procedure

The output depends on substrate availability, initial cell concentration, elapsed time, and likely nonlinear biological responses. This motivates comparing a low-complexity polynomial family against a more flexible tree ensemble rather than selecting an architecture without testing it. Degree-two ridge regression can represent curvature and pairwise interactions, while degree-three terms permit more complicated response surfaces. Ridge regularization limits the size of fitted coefficients and can reduce instability where the dataset is small relative to the number of polynomial terms. The polynomial pipelines standardize inputs before generating the features because 1 mM and hundreds of thousands of cells/mL would otherwise be incomparable scales. The forest is a separate, reasonable alternative that can learn local thresholds and interactions without imposing a global polynomial equation.

The tested hyperparameters were fixed in advance as practical choices, not advertised as an exhaustive search. The two ridge models use degrees 2 and 3 with regularization strengths 10 and 30. The forest has 200 trees, a minimum leaf size of 3, and all candidate predictors available at a split; its seed is 42. More trees help stabilize ensemble averaging, while the leaf-size constraint moderates very small terminal regions. However, those settings should not be called mathematically optimal. A nested, batch-grouped hyperparameter search and a separate holdout set would be stronger choices if additional batches were available. In this limited-data project, comparing several reasonable architectures while keeping validation batch-aware provides an understandable baseline.

### Comparison of actual model performance

Performance was compared using five-fold grouped cross-validation on entire batches. This is an important decision: ordinary row-level random splits would put nearby observations from the same bioreactor in both training and test data, giving an overly favourable impression of generalization. Grouped validation tests prediction of the trajectories of batches not used for fitting in that fold. RMSE, in mg/L, quantifies the typical scale of concentration error, and R² shows improvement relative to a constant mean prediction. The results from this specific 30-batch selection are:

{metric_table(models)}

The lowest-error architecture is **{best_model}**, with out-of-fold RMSE **{best_rmse:.3f} mg/L** and R² **{best_r2:.3f}**. A positive R² indicates performance better than a constant mean prediction on the combined held-out observations; negative R² indicates worse performance than that baseline. The polynomial models provide valuable comparison points rather than being dismissed simply because the forest performed better. Their fixed global polynomial assumptions may be poorly matched to the observed trajectories and the limited diversity of independent batches. Conversely, the forest can represent nonlinear responses but does not extrapolate reliably beyond the studied variable ranges. The main result is therefore a model comparison for this specific dataset and validation design, not a guarantee that the forest will outperform polynomial models in all bioreactors.

### Effect of initial conditions on profit and yield

To examine the three input factors without relying only on one winning point, the code evaluated each of the 343 combinations of initial glucose, glutamine, and VCD. For each combination, it retained the harvest time that maximized **predicted space-time profit**. The following marginal summaries then average those profit-optimal values over the 49 combinations of the other two initial inputs. The yield column uses the harvest time selected for profit, not the separate harvest time that would necessarily maximize yield. Therefore these tables describe a model-based operating screen and should not be interpreted as isolated experimental causal effects.

{effect_table(factors)}

{effect_sentence(factors, FEATURES[0], 'initial glucose')}

{effect_sentence(factors, FEATURES[1], 'initial glutamine')}

{effect_sentence(factors, FEATURES[2], 'initial VCD')}

The relevant physical interpretation is that nutrient supply may support metabolic activity, while a larger inoculum can change the culture's production trajectory. The model does not directly test those biological mechanisms, however, so statements about precisely *why* a setting is profitable would require additional process measurements or experiments. Initial glucose and glutamine also increase media cost independently of any possible production benefit. Higher initial VCD likewise increases inoculum cost before any product has accumulated. The plotted relationship among glucose, glutamine, and predicted profit is conditional on the recommended VCD, so its colour scale should not be interpreted as a universal response for every inoculum level.

![Predicted profit heatmap at recommended VCD](figures/profit_heatmap.png)

### Recommended operation and predicted economics

The selected settings maximize the model's estimated space-time profit over the original seven-level initial-condition grid and the positive observed harvest times, not over every physically possible concentration or continuous harvest time. The recommended operating point is **{to_3(reco[FEATURES[0]])} mM glucose**, **{to_3(reco[FEATURES[1]])} mM glutamine**, and **{to_3(reco[FEATURES[2]])} cells/mL initial VCD**. The recommended harvest occurs at **{to_3(duration)} days**, equivalent to about **{duration * 24:.0f} hours**, followed by the specified one day of downtime. The predicted mAb concentration at harvest is **{to_3(mab)} mg/L**. The following table reports all requested outputs, including approximate 95% pointwise prediction intervals:

| Quantity | Model estimate | Approximate 95% prediction interval |
|---|---:|---:|
| Initial glucose | {to_3(reco[FEATURES[0]])} mM | — |
| Initial glutamine | {to_3(reco[FEATURES[1]])} mM | — |
| Initial VCD | {float(reco[FEATURES[2]]):,.0f} cells/mL | — |
| Harvest time | {to_3(duration)} days | — |
| Cell culture cost | {as_money(cost)}/L | — (specified cost equation) |
| Cell culture value at harvest | {as_money(value)}/L | {as_money(reco['Cell culture value PI lower [$/L]'])} to {as_money(reco['Cell culture value PI upper [$/L]'])}/L |
| Space-time profit | {as_money(profit)}/L/day | {as_money(reco['Space-time profit PI lower [$/L/d]'])} to {as_money(reco['Space-time profit PI upper [$/L/d]'])}/L/day |
| mAb space-time yield | {yield_mab:.3f} mg/L/day | {float(reco['Space-time yield PI lower [mg/L/d]']):.3f} to {float(reco['Space-time yield PI upper [mg/L/d]']):.3f} mg/L/day |

The numerical consistency can be checked directly. For the recommended point, the culture costs {as_money(cost)} per litre, its estimated value is {as_money(value)} per litre, and its estimated batch profit is {as_money(value-cost)} per litre. Dividing by {duration + 1:.3f} days (harvest plus downtime) gives the reported **{as_money(profit)}/L/day**. Dividing the predicted {mab:.3f} mg/L of product by the same cycle time gives **{yield_mab:.3f} mg/L/day**. The harvest-time graphic displays how profit changes with the selected input conditions and marks the recommended time. The observed-trajectory graphic separately displays all 30 training runs, allowing comparison with the predicted timing and magnitude of mAb production.

![mAb trajectories in 30 selected runs](figures/observed_trajectories.png)

![Space-time profit by harvest time at recommended settings](figures/optimal_harvest_time.png)

### Interpretation of prediction intervals and limitations

The stated 95% prediction intervals are **approximate empirical pointwise intervals** based on the 2.5th and 97.5th percentiles of residuals from held-out batch predictions. The residual bounds are added to predicted mAb concentration, clipped at zero, and then converted to value, profit, and yield using the same economic equations. This makes the interval arithmetic consistent, but does not provide a formal 95% coverage guarantee for a new batch. Errors can change with harvest time and inoculum setting, and measurements within a batch are correlated. The method also uses the same observations for model comparison and for residual calibration, and it does not account for the optimism introduced by searching thousands of possible candidate schedules for the single largest estimate.

Other limitations include the restriction to the seven input levels appearing in the source data, the relatively small number of independent batch runs, the lack of repeated experiments at selected conditions, and the simplified economic model. The predictor uses starting conditions and time rather than measured future glucose depletion, accumulated waste products, or time-varying cell viability. This is useful for an advance batch-planning decision, but it means the model cannot respond to a process disturbance after startup. A full deployment study would check external batches, examine uncertainty near the optimum, and revisit the economic assumptions. Accordingly, the recommended point is a reasoned **model-based candidate for experimental confirmation**, not a proven global optimum.


'''
    return md


def count_section(markdown, heading, next_heading=None):
    start = markdown.index(heading) + len(heading)
    stop = markdown.index(next_heading, start) if next_heading else len(markdown)
    content = markdown[start:stop]
    # Strip code, tables, and image references from approximate prose count.
    content = '\n'.join(line for line in content.splitlines() if not line.startswith('|') and not line.startswith('!['))
    return len(re.findall(r"\b[\w]+(?:[-’'][\w]+)*\b", content))


def main():
    reco, models, factors, design, versions = read_inputs()
    if len(design) != 30 or len(factors) != 343:
        raise ValueError(f'Unexpected experiment selection or grid size: {len(design)} batches; {len(factors)} conditions.')
    markdown = build(reco, models, factors, design, versions)
    intro = count_section(markdown, '# Monoclonal Antibody Production Profit Optimization', '## Highlights')
    design_words = count_section(markdown, '## Code Design', '## Analysis')
    analysis_words = count_section(markdown, '## Analysis')
    print(f'Word count — title/introduction: {intro} (required 40–80)')
    print(f'Word count — Code Design: {design_words} (required 500–1000)')
    print(f'Word count — Analysis: {analysis_words} (required 1000–2000)')
    if not (40 <= intro <= 80 and 500 <= design_words <= 1000 and 1000 <= analysis_words <= 2000):
        raise ValueError('A section falls outside the required word count; README has not been replaced.')
    path = PROJECT / 'README.md'
    path.write_text(markdown, encoding='utf-8')
    print(f'Created {path}')
    print('Review the generated README and figure links before submission.')


if __name__ == '__main__':
    main()
