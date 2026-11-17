# RailPulse AI Dashboard

## 1. Put the files in the project root

Copy `app.py` and `requirements_dashboard.txt` into:

`C:\Users\Login\Documents\RailPulse_AI\`

The project should look like:

```text
RailPulse_AI/
├── app.py
├── requirements_dashboard.txt
├── data/
├── notebooks/
└── outputs/
    ├── tables/
    └── figures/
```

The dashboard reads the project's existing CSV result files from `outputs/tables/`.
It intentionally does not load the 1.1-million-row feature dataset at startup.

## 2. Install dashboard dependencies

Open Command Prompt or Anaconda Prompt, activate the Python environment used for the project, then run:

```bash
cd /d C:\Users\Login\Documents\RailPulse_AI
python -m pip install -r requirements_dashboard.txt
```

## 3. Launch

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`.

## 4. Dashboard sections

- Command Center
- Delay Intelligence
- Model Performance
- Pattern Discovery
- Explainability & Data Quality

## Notes

- This is a historical research dashboard, not a live Indian Railways feed.
- It reads actual experiment outputs; missing CSVs are handled gracefully.
- The selected model's reported metrics are the saved reserved-group results.
- The reserved train group had been examined in earlier exploratory work; the dashboard discloses this limitation.
- If a table filename changes, update the corresponding `load_table()` call in `app.py`.
