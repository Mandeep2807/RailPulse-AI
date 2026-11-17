

\# 🚆 RailPulse AI



\### Intelligent Next-Station Train Delay Prediction and Route Reliability Analysis



RailPulse AI is a student Predictive Analytics project that uses historical Indian Railways operational data to estimate the arrival delay of a train at its next station.



\## 🎯 Project Objective



To study railway delay patterns and predict next-station arrival delays using machine learning.



\## 📊 Dataset



\- Historical data: September 2024

\- Initial delay observations: 1,282,325

\- Final paired observations: 1,101,443

\- Unique trains in paired dataset: 3,146



The original data was obtained from the IIT Kharagpur RSTGCN research repository.



Raw datasets are not included in this repository. Refer to the original source and its terms before obtaining or redistributing the data.



\## ⚙️ Technologies Used



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- SQLite

\- Jupyter Notebook

\- Streamlit

\- Plotly

\- Matplotlib



\## 🤖 Machine Learning



The project compares multiple regression approaches, including Linear Regression, Random Forest and HistGradientBoosting.



The selected model uses a four-feature Delta Learning approach with HistGradientBoostingRegressor.



\### Final evaluation results



| Metric | Result |

|---|---:|

| MAE | 9.4939 minutes |

| MSE | 846.8970 |

| RMSE | 29.1015 minutes |

| R² | 0.8628 |



These results come from the reserved train-group evaluation. The group was examined during earlier exploratory experiments and is not claimed to be a completely untouched test set.



\## 📈 Dashboard



The Streamlit dashboard presents:

\- Project overview

\- Historical delay analysis

\- Model performance

\- Feature importance, PCA, clustering and data-quality findings



\## ▶️ Run the Dashboard



Install the dependencies:



```bash

pip install -r requirements\_dashboard.txt

```



Run:



```bash

streamlit run app.py

```



The dashboard uses selected result CSV files in `outputs/tables/` and does not provide live train tracking.



\## 📁 Project Structure



```text

RailPulse\_AI/

├── app.py

├── requirements\_dashboard.txt

├── notebooks/

├── outputs/

│   ├── figures/

│   └── tables/

├── data/

└── README.md

```



\## ⚠️ Limitations



\- The project uses historical September 2024 data.

\- It is a research prototype, not a live railway prediction service.

\- Prediction performance is not uniform across delay categories.

\- The reserved evaluation group was examined during exploratory work.

\- Dataset redistribution rights must be checked against the original source.



\## 👨‍💻 Project



Developed as an academic Predictive Analytics project.



