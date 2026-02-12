# HR Analytics: Employee Turnover Predictor

Streamlit dashboard that predicts employee turnover risk using `turnover.csv`.

## Features
- Employee data import (`CSV`/`Excel`) or manual employee input
- ML turnover risk prediction (compares Logistic Regression and Random Forest)
- Risk score dashboard with visualizations
- Filters by department, tenure, and risk level
- Actionable retention recommendations for at-risk employees

## Dataset
Default path used by app:
- `data/turnover.csv`

You can also upload your own `CSV`/`Excel` file from the sidebar.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## React Frontend Prototype
A separate React UI is available in `frontend/` for a cleaner product-style experience.

Run:
```bash
cd frontend
npm install
npm run dev
```

## Notes
- Target column is expected to be `event` (1 = turnover, 0 = stayed).
- The app auto-handles non-UTF8 CSV encoding with a latin1 fallback.
