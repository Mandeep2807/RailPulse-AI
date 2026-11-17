# 🚆 RailPulse AI
### Understanding train delays, one station at a time.

**An end-to-end predictive analytics project for next-station train delay prediction and railway route reliability analysis using historical Indian Railways operational data.**

[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Data%20Analysis-Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Project-Research%20Prototype-0A7B83)]()

---

## The idea behind RailPulse AI

A train may leave one station a few minutes late and reach the next station with a completely different delay. Sometimes the delay grows. Sometimes it remains nearly unchanged. And sometimes the train makes up time along the route.

That changing delay is what makes railway operations interesting from a predictive analytics perspective.

**RailPulse AI explores a simple but important question:**

*Can we estimate how late a train will arrive at its next station using information available at its current station?*

Instead of treating an entire train journey as one prediction problem, this project looks at consecutive station transitions. It studies how delays behave between stations, compares different machine learning approaches, and investigates whether useful patterns can be identified across trains and routes.

The result is a research-oriented predictive analytics system supported by an interactive dashboard.

> This project uses historical data. It is not a live train-tracking system and does not provide official railway arrival predictions.

## Explore the project

| Resource | Link |
|---|---|
| Live dashboard | [Open RailPulse AI Dashboard](https://railpulse-ai-rwhgcgjhzqqdxbnayvftbd.streamlit.app/) |
| Source code | [GitHub Repository](https://github.com/Mandeep2807/RailPulse-AI) |
| Dataset | [Indian Railway Network and Delays – RSTGCN](https://github.com/KoyenaChowdhury/RSTGCN) |

*If the dashboard link is not active yet, it will become available after deployment.*

---

## What does the project actually do?

RailPulse AI follows an end-to-end workflow, starting with raw operational data and ending with model evaluation and an interactive presentation of the findings.

The main components are:

- **Data auditing:** Examine train delay records, station information and route sequences.
- **Data engineering:** Match observations from consecutive stations to create next-station prediction examples.
- **Exploratory analysis:** Understand delay distributions, daily patterns and relationships between consecutive delays.
- **Feature engineering:** Build meaningful predictors from information available at the current station.
- **Model development:** Compare regression baselines and machine learning models.
- **Dimensionality reduction and clustering:** Use PCA and K-Means to explore patterns in route and delay behaviour.
- **Model evaluation:** Examine prediction errors, generalization across trains, feature importance and sensitivity to data-quality issues.
- **Dashboard:** Present the results through an interactive Streamlit application.

## The dataset

The project uses the **Indian Railway Network and Delays** dataset made available through the RSTGCN research repository by Koyena Chowdhury and collaborators.

Source: [KoyenaChowdhury/RSTGCN](https://github.com/KoyenaChowdhury/RSTGCN)

The source collection contains railway network connectivity, station-to-zone mappings, train routes and observed operational delays. Our project primarily works with the September 2024 train-level route and delay records.

### Dataset coverage used in this project

| Description | Count |
|---|---:|
| Observation period | 1–30 September 2024 |
| Train records in delay data | 3,892 unique trains |
| Stations represented in the delay data | 4,736 |
| Original delay observations | 1,282,325 |
| Final consecutive-station observations | 1,101,443 |
| Unique trains in the paired modelling dataset | 3,146 |
| Observation dates | 30 |

These figures describe the data processed in this project. They should not be interpreted as a claim that every Indian train or every railway station was observed during the period.

### How the raw data was used

The source data includes:

- `IRN_edges.csv` — railway network connectivity.
- `stations_zones_mapping.json` — station-to-zone mapping.
- `train_delays_Sep2024.json` — nested train, date and station delay observations.
- `train_routes_delays_Sep2024.csv` — flattened train-level operational delay records.
- `train_routes_Sep2024.csv` — scheduled train routes and station sequences.

The route and delay information was combined to identify consecutive station observations belonging to the same train and journey date.

This step produced the paired modelling dataset used throughout the prediction experiments.

### Data quality was part of the research

The project did not assume that every record was correct simply because it existed in the source files.

The data preparation process included:

- Checking duplicate records and route sequences.
- Investigating missing and unmatched observations.
- Auditing consecutive station transitions.
- Examining timestamp inconsistencies.
- Investigating unusually large delay discrepancies.
- Avoiding automatic corrections where the original timestamp information was ambiguous.

The final paired dataset contains **1,101,443 valid current-to-next-station observations**.

Raw datasets and intermediate processing files are not included in this repository because of their size and to keep the public project repository manageable. Their source is linked above.

---

## How next-station prediction works

Each modelling observation represents one transition:

**Current station → Next station**

At the current station, the system uses four predictors:

| Feature | Meaning |
|---|---|
| `current_arr_delay` | Train's arrival delay at its current station |
| `distance_to_next` | Distance from the current station to the next station |
| `current_distance` | Route distance covered up to the current station |
| `current_sequence` | Current station's position in the train's route |

The prediction target is:

`next_arr_delay`

This represents the train's arrival delay at the next station, measured in minutes.

An important design decision was to avoid using information from the future station as an input to its own prediction. This helps maintain a realistic prediction setup and reduces the risk of target leakage.

## The modelling approach

Rather than selecting a complicated algorithm immediately, we started with simpler approaches and gradually compared more capable models.

### Models explored

| Model | Why it was included |
|---|---|
| Dummy Regressor | Establishes a simple reference point |
| Persistence Baseline | Tests whether the current delay can serve as a useful next-station estimate |
| Linear Regression | Provides an interpretable regression benchmark |
| Random Forest | Captures nonlinear relationships through an ensemble of decision trees |
| HistGradientBoosting | Tests a gradient-boosted tree approach |
| Reduced Delta Learning | Learns the change in delay instead of predicting the next delay directly |

### Why Reduced Delta Learning?

One useful observation from the exploratory analysis was that the current and next station delays are strongly related.

Instead of directly estimating the next delay, we experimented with predicting the change in delay between consecutive stations.

The idea is:

**Predicted next delay = Current delay + Predicted delay change**

The final approach uses HistGradientBoostingRegressor to estimate that change.

Predictions are constrained to be non-negative because the target represents arrival delay in minutes.

The final model uses:

- Absolute error loss.
- Learning rate: 0.06.
- Maximum iterations: 100.
- Maximum leaf nodes: 31.
- Minimum samples per leaf: 40.
- L2 regularization: 2.0.
- Early stopping enabled.
- Random state: 42.

The final feature set contains four predictors selected after feature comparison and ablation experiments.

---

## Evaluation strategy

A single random train-test split can make a predictive model look more reliable than it really is, especially when observations from the same train appear in both training and testing data.

For this reason, the project explored multiple evaluation strategies.

### 1. Chronological evaluation

The model development process used earlier September observations for training and later September observations for testing.

This helped examine how the model behaved when evaluated on a later period.

### 2. Unseen-train evaluation

Train identities were separated between training and evaluation groups.

This helped investigate whether the learned patterns transferred to trains that were not present in the training group.

### 3. Three-way train-level evaluation

A further train-level split separated:

- Internal training trains.
- Validation trains.
- A reserved evaluation group.

The final comparison was performed using the reserved group.

**Research transparency:** The reserved evaluation group was examined in earlier exploratory experiments. It is therefore described as a reserved train-group evaluation, not a completely untouched final test set.

### Evaluation metrics

Since the main task is regression, the project uses the following metrics:

- **MAE:** Average absolute prediction error.
- **MSE:** Mean squared prediction error, which gives greater weight to larger errors.
- **RMSE:** Square root of MSE, expressed in the same unit as the target.
- **R²:** Measures the proportion of target variance explained relative to a mean-prediction reference.

All delay errors are measured in minutes.

---

## Final model results

The final comparison uses the same reserved train-group evaluation observations for both approaches.

| Metric | Persistence Baseline | Reduced Delta Learning |
|---|---:|---:|
| MAE | 9.6001 | **9.4939** |
| MSE | 853.1241 | **846.8970** |
| RMSE | 29.2083 | **29.1015** |
| R² | 0.8618 | **0.8628** |

### What do these results tell us?

Reduced Delta Learning achieved a lower error than the Persistence Baseline across all four reported metrics.

The MAE improved by approximately **0.1062 minutes**, equivalent to an improvement of approximately **1.11%** over the baseline.

This is a modest improvement, not a dramatic breakthrough.

The result is still useful because it shows that learning the change in delay can provide an incremental improvement over simply carrying the current delay forward.

The project also examines how model performance varies across delay categories. The final model does not outperform the persistence approach uniformly in every category, which is an important part of interpreting the results.

### A note about prediction errors

The target distribution contains a large number of low-delay observations and a comparatively small number of very large delays.

Consequently, a model can perform reasonably well on average while still making substantial errors on unusual cases.

The residual analysis identified extreme errors approaching a full day in magnitude. These cases are important limitations and are not hidden by the overall metrics.

---

## Beyond prediction: PCA and clustering

RailPulse AI also investigates whether the operational data contains useful patterns beyond the prediction task.

### Principal Component Analysis (PCA)

PCA was applied to explore a lower-dimensional representation of the engineered features.

For the four-feature analysis:

- PC1 explained approximately 46.31% of the variance.
- PC2 explained approximately 26.36%.
- Together, the first two components explained approximately **72.67%** of the variance.

This provides a compact visual representation of the four-dimensional feature space.

### K-Means clustering

K-Means was used to explore groups of observations with similar feature patterns.

The project examined cluster-selection metrics and cluster profiles rather than choosing the number of clusters arbitrarily.

A separate delay-focused clustering experiment used current delay and distance to the next station. Six clusters were selected from the candidate configurations examined.

These clusters are descriptive groups for analysis. They should not be interpreted as official railway categories or proof of causal delay mechanisms.

---

## Model explainability and robustness

A predictive result is more useful when we can understand which inputs influence it and whether the result is sensitive to data-quality decisions.

### Feature importance

Permutation importance was calculated for the final four-feature model.

The importance scores represent the increase in MAE after shuffling a feature on the validation sample. They are not percentage shares of model influence.

The results indicate that current arrival delay is the most influential predictor, while distance to the next station, current route distance and route sequence also contribute.

### Feature ablation

Feature ablation experiments compared different feature combinations to check whether additional inputs actually improved validation performance.

The reduced four-feature configuration was retained for the final model.

### Sensitivity experiment

The project also investigated observations affected by major timestamp discrepancies.

A sensitivity experiment excluded 183 affected modelling rows and compared the resulting metrics with the original evaluation.

The resulting performance difference was small:

| Metric | Original evaluation | Sensitivity experiment |
|---|---:|---:|
| MAE | 9.4939 | 9.5034 |
| MSE | 846.8970 | 847.3491 |
| RMSE | 29.1015 | 29.1093 |
| R² | 0.8628 | 0.8627 |

This suggests that the overall evaluation metrics were relatively stable under this particular exclusion experiment. It does not establish that all data-quality issues have been eliminated.

---

## The RailPulse AI dashboard

The project includes a Streamlit dashboard designed to make the analysis easier to explore without requiring users to read every notebook.

The dashboard is organised into four sections:

| Section | What it covers |
|---|---|
| Overview | Project introduction and key dataset insights |
| Delay Analysis | Delay patterns and descriptive analysis |
| Model Results | Model comparisons and predictive performance |
| What We Learned | Important findings, feature contributions and limitations |

The dashboard presents saved project outputs. It does not retrain the model or connect to live railway operational systems.

---

## Project workflow

The work is organised into twelve Jupyter notebooks.

| Notebook | Main responsibility |
|---|---|
| `01_dataset_audit.ipynb` | Dataset inspection and quality auditing |
| `02_feature_engineering.ipynb` | Creation of modelling features |
| `03_split_validation.ipynb` | Train-level data splitting |
| `04_baseline_models.ipynb` | Initial regression baselines |
| `05_model_validation.ipynb` | Model comparison and validation |
| `06_model_explainability.ipynb` | Explainability analysis |
| `07_feature_selection.ipynb` | Feature ablation and selection |
| `08_pca_clustering.ipynb` | PCA and K-Means experiments |
| `09_final_model_evaluation.ipynb` | Final model evaluation |
| `10_data_quality_correction.ipynb` | Timestamp and data-quality investigation |
| `11_sensitivity_experiment.ipynb` | Sensitivity analysis |
| `12_final_results_consolidation.ipynb` | Consolidation of final results |

### Repository structure

```text
RailPulse-AI/
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_split_validation.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_model_validation.ipynb
│   ├── 06_model_explainability.ipynb
│   ├── 07_feature_selection.ipynb
│   ├── 08_pca_clustering.ipynb
│   ├── 09_final_model_evaluation.ipynb
│   ├── 10_data_quality_correction.ipynb
│   ├── 11_sensitivity_experiment.ipynb
│   └── 12_final_results_consolidation.ipynb
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── app.py
├── requirements.txt
├── requirements_dashboard.txt
├── DASHBOARD_README.md
├── README.md
├── README.txt
└── .gitignore
```

The raw and intermediate datasets are maintained separately and are not included in the public repository.

---

## Running the dashboard locally

### 1. Clone the repository

```bash
git clone https://github.com/Mandeep2807/RailPulse-AI.git
cd RailPulse-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dashboard dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the dashboard

```bash
streamlit run app.py
```

The dashboard should open in your browser at the local Streamlit address displayed in the terminal.

**Note:** The root `requirements.txt` is intended for the dashboard. Reproducing all notebook experiments may require additional scientific Python packages and access to the original and processed datasets.

---

## Tools and technologies

| Area | Technologies |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualisation | Matplotlib, Plotly |
| Dimensionality reduction | PCA |
| Clustering | K-Means |
| Interactive dashboard | Streamlit |
| Development | JupyterLab |
| Version control | Git and GitHub |

---

## Limitations and future directions

This project is a historical research prototype, and its results should be interpreted within the limits of the available data and experiments.

### Current limitations

- The operational observations cover September 2024 rather than continuous railway activity.
- The final model uses a compact set of four predictors.
- Extreme delay observations can produce substantial prediction errors.
- The final reserved train-group evaluation was not completely untouched because it was examined in earlier exploratory work.
- The project does not model the full railway network as a live, continuously updated system.
- Performance on the available sample does not establish generalization to every Indian train, route or operating condition.

### Possible future improvements

- Incorporate additional months and more recent operational observations.
- Investigate weather, congestion, timetable and external operational factors where reliable data is available.
- Explore graph-based and spatio-temporal models for railway network dependencies.
- Evaluate stronger temporal and route-aware validation strategies.
- Improve prediction intervals and uncertainty estimation.
- Study delay propagation and recovery over longer station sequences.
- Develop a carefully validated live-data integration if suitable official data access becomes available.

---

## Research references and dataset acknowledgement

This project builds on publicly described railway research data and is an independent predictive analytics implementation.

**Dataset source**

Koyena Chowdhury et al., Indian Railway Network and Delays dataset.

[Source repository – RSTGCN](https://github.com/KoyenaChowdhury/RSTGCN)

**Related research**

K. Chowdhury, P. Koley, A. Chakraborty and S. Ghosh, *RSTGCN: Railway-centric Spatio-Temporal Graph Convolutional Network for Train Delay Prediction*, IEEE Transactions on Intelligent Transportation Systems (2026).

[arXiv preprint: 2510.01262](https://arxiv.org/abs/2510.01262)

The related paper and dataset are acknowledged as the source of the underlying railway data. RailPulse AI uses a different modelling formulation focused on individual consecutive-station delay prediction.

**Dataset usage note:** Please consult the original dataset repository and its current terms before redistributing source data. This project does not claim ownership of the original railway records or grant a licence to redistribute them.

---

## A final note

RailPulse AI began with a practical question about train delays and developed into a complete predictive analytics workflow.

The most meaningful part of the work was not simply obtaining a model score. It was examining the data carefully, comparing simple and advanced approaches, testing generalization across trains, investigating unusual records, and understanding where the predictions work and where they still fall short.

The project is an attempt to turn historical railway observations into something that can be explored, evaluated and understood.

**Built as an academic Predictive Analytics project for INT234.**

*RailPulse AI — Understanding the journey, one station at a time.*
