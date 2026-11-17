# RailPulse AI — Simple Dashboard

This version is intentionally designed to be easy to understand and explain in a university presentation.

## Install
Place `app.py` and `requirements_dashboard.txt` in:
`C:\Users\Login\Documents\RailPulse_AI\`

Keep the existing `outputs/tables/` folder beside them.

Run in Command Prompt:
```bash
cd /d C:\Users\Login\Documents\RailPulse_AI
python -m pip install -r requirements_dashboard.txt
python -m streamlit run app.py
```

## Four simple pages
1. Overview — project KPIs, daily trend and delay categories.
2. Delay Analysis — date filter and delay distribution.
3. Model Results — baseline comparison and plain-language metric explanations.
4. What We Learned — feature importance, PCA, clusters and data-quality sensitivity.

This is a historical research dashboard, not live train tracking. It reads the saved project result CSVs and does not load the full 1.1M-row modelling file at startup.

## Updated feature-importance chart
The dashboard now reads `outputs/tables/final_model_permutation_importance.csv`,
which was calculated for the selected four-feature model on a 30,000-row
internal-validation sample. The chart uses plain-language feature labels.
