# HR Analytics Frontend (React)

Personal-project React frontend for turnover risk exploration.

## What it includes
- Plain-English labels (no raw abbreviations in UI)
- Tabbed UX: Overview, At-Risk, Scenario, Guide
- Sidebar filters (department, risk level, tenure)
- KPI cards, risk distribution bars, department ranking
- High-risk employee table with action hints
- Scenario predictor with live risk estimation

## Run locally
```bash
cd frontend
npm install
npm run dev
```

Then open the local Vite URL shown in terminal.

## Build
```bash
npm run build
npm run preview
```

## Data source
- Loads CSV from `frontend/public/data/turnover.csv`
