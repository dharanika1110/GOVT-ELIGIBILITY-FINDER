# Know Your Scheme – Government Scheme Eligibility Finder

Streamlit app: rule-based eligibility engine + match score, eligibility classifier
(Logistic Regression vs Random Forest), and NFHS-5 district analytics (K-Means + Random Forest regression).

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this folder to GitHub (keep `data/` and `.streamlit/`).
2. share.streamlit.io -> New app -> pick the repo -> main file `app.py` -> Deploy.

## Files
- `app.py` – UI (6 tabs, incl. 'Analyze Your Data' workbench)  - `engine.py` – scheme data, rules, ML, NFHS analytics
- `analytics.py` – generic data workbench (any CSV/Excel/JSON: cleaning, clusters, drivers, priority ranking, scheme gap)
- `data/schemes_processed.csv` (3,397 myScheme schemes), `data/nfhs_district_wide.csv` (NFHS-5 districts) – replace with updated files of the same columns to refresh

## Folder layout
```
app.py  engine.py  analytics.py  requirements.txt  README.md
.streamlit/config.toml
data/schemes_processed.csv
data/nfhs_district_wide.csv
```
