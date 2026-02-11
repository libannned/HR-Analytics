import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "event"
TENURE_COL = "stag"
INDUSTRY_COL = "industry"


def read_table(file_or_path):
    if isinstance(file_or_path, str):
        if file_or_path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(file_or_path)
        try:
            return pd.read_csv(file_or_path)
        except UnicodeDecodeError:
            return pd.read_csv(file_or_path, encoding="latin1")
    name = file_or_path.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_or_path)
    try:
        return pd.read_csv(file_or_path)
    except UnicodeDecodeError:
        file_or_path.seek(0)
        return pd.read_csv(file_or_path, encoding="latin1")


@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return read_table(uploaded_file)

    default_paths = [
        "data/turnover.csv",
        "turnover.csv",
        "/Users/najmamahad/Downloads/turnover.csv",
    ]
    for path in default_paths:
        try:
            return read_table(path)
        except FileNotFoundError:
            continue

    raise FileNotFoundError("Could not find turnover.csv. Upload a file in the sidebar.")


@st.cache_resource(show_spinner=False)
def train_models(df):
    data = df.copy()

    if TARGET_COL not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    data = data.dropna(subset=[TARGET_COL])
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data = data.dropna(subset=[TARGET_COL])
    data[TARGET_COL] = data[TARGET_COL].astype(int)

    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        ),
    }

    scores = []
    fitted_models = {}
    for model_name, model in candidates.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        metrics = {
            "model": model_name,
            "roc_auc": roc_auc_score(y_test, prob),
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        }
        scores.append(metrics)
        fitted_models[model_name] = pipe

    score_df = pd.DataFrame(scores).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_name = score_df.iloc[0]["model"]

    best_model = fitted_models[best_name]
    all_prob = best_model.predict_proba(X)[:, 1]
    scored_df = data.copy()
    scored_df["risk_score"] = all_prob
    scored_df["risk_level"] = pd.cut(
        scored_df["risk_score"],
        bins=[-0.01, 0.4, 0.7, 1.01],
        labels=["Low", "Medium", "High"],
    )

    return {
        "score_df": score_df,
        "best_name": best_name,
        "best_model": best_model,
        "scored_df": scored_df,
        "feature_cols": X.columns.tolist(),
    }


def retention_recommendations(row, reference_df):
    recs = []

    if "anxiety" in row and "anxiety" in reference_df:
        if pd.to_numeric(row["anxiety"], errors="coerce") >= reference_df["anxiety"].quantile(0.75):
            recs.append("High anxiety signal: schedule manager check-ins and wellbeing support.")

    if "selfcontrol" in row and "selfcontrol" in reference_df:
        if pd.to_numeric(row["selfcontrol"], errors="coerce") <= reference_df["selfcontrol"].quantile(0.25):
            recs.append("Low self-control score: provide clear weekly goals and structured task planning.")

    if TENURE_COL in row and TENURE_COL in reference_df:
        if pd.to_numeric(row[TENURE_COL], errors="coerce") <= reference_df[TENURE_COL].quantile(0.25):
            recs.append("Short tenure risk: reinforce onboarding, buddy support, and early-career coaching.")

    if "coach" in row and str(row["coach"]).strip().lower() == "no":
        recs.append("No coaching support: assign mentor/coach for the next 60 days.")

    if "greywage" in row and str(row["greywage"]).strip().lower() == "grey":
        recs.append("Compensation concern: review pay transparency and fairness with HR.")

    if not recs:
        recs.append("General retention action: run a 1:1 stay interview and create a 30-day engagement plan.")

    return recs


def render_risk_interpretation_guide():
    with st.expander("How To Read Risk Scores", expanded=False):
        st.markdown(
            """
**Risk Levels**
- `Low` (`< 0.40`): Employee is currently unlikely to leave soon.
- `Medium` (`0.40 to 0.69`): Warning signs are present. Early intervention is recommended.
- `High` (`>= 0.70`): Strong turnover signal. Immediate retention action is recommended.

**How To Interpret A Score**
- `0.15` (15%): low concern, maintain normal engagement.
- `0.52` (52%): moderate concern, start targeted check-ins.
- `0.84` (84%): high concern, create a concrete retention plan now.

**Recommended Actions By Level**
1. Low:
   Keep regular 1:1 cadence, reinforce growth path and recognition, recheck monthly.
2. Medium:
   Run a stay interview within 1 to 2 weeks, identify top friction points, and set a 30-day plan.
3. High:
   Start manager plus HR intervention in 48 to 72 hours, build a formal retention plan, and follow up weekly.

**Factor-Based Translation**
- High `anxiety`: potential stress/support issue.
- Low `selfcontrol`: may need clearer goals and structured planning.
- Short `stag` (tenure): onboarding/early-tenure risk.
- `coach = no`: add mentor/coach support.
- `greywage = grey`: possible compensation trust/fairness issue.

**Important Note**
- This model is an early-warning signal, not proof. Confirm with manager context and direct employee conversation.
"""
        )


def main():
    st.set_page_config(page_title="HR Analytics: Turnover Predictor", layout="wide")
    st.title("HR Analytics: Employee Turnover Predictor")
    st.caption(
        "Predict turnover risk, identify at-risk employees, filter by department/tenure/risk level, and generate retention actions."
    )

    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload employee data (CSV/Excel)", type=["csv", "xlsx", "xls"]
    )

    try:
        raw_df = load_data(uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.success(f"Loaded {len(raw_df):,} rows and {len(raw_df.columns)} columns")

    try:
        model_bundle = train_models(raw_df)
    except Exception as exc:
        st.error(f"Training failed: {exc}")
        st.stop()

    score_df = model_bundle["score_df"]
    best_name = model_bundle["best_name"]
    best_model = model_bundle["best_model"]
    scored_df = model_bundle["scored_df"].copy()

    st.subheader("Model Performance")
    st.dataframe(score_df.style.format({c: "{:.3f}" for c in score_df.columns if c != "model"}), use_container_width=True)
    st.info(f"Best model by ROC-AUC: {best_name}")

    st.subheader("Risk Score Dashboard")
    render_risk_interpretation_guide()

    departments = sorted(scored_df[INDUSTRY_COL].dropna().astype(str).unique()) if INDUSTRY_COL in scored_df.columns else []
    tenure_min = float(scored_df[TENURE_COL].min()) if TENURE_COL in scored_df.columns else 0.0
    tenure_max = float(scored_df[TENURE_COL].max()) if TENURE_COL in scored_df.columns else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_departments = st.multiselect("Filter by department", departments, default=departments)
    with c2:
        selected_risk = st.multiselect("Filter by risk level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    with c3:
        selected_tenure = st.slider(
            "Filter by tenure",
            min_value=tenure_min,
            max_value=tenure_max,
            value=(tenure_min, tenure_max),
        )

    filtered = scored_df.copy()
    if selected_departments and INDUSTRY_COL in filtered.columns:
        filtered = filtered[filtered[INDUSTRY_COL].astype(str).isin(selected_departments)]
    if selected_risk:
        filtered = filtered[filtered["risk_level"].astype(str).isin(selected_risk)]
    if TENURE_COL in filtered.columns:
        filtered = filtered[
            (filtered[TENURE_COL] >= selected_tenure[0]) & (filtered[TENURE_COL] <= selected_tenure[1])
        ]

    k1, k2, k3 = st.columns(3)
    k1.metric("Employees (Filtered)", f"{len(filtered):,}")
    k2.metric("At-Risk (High)", f"{(filtered['risk_level'].astype(str) == 'High').sum():,}")
    k3.metric("Avg. Risk Score", f"{filtered['risk_score'].mean():.2f}" if len(filtered) else "N/A")

    v1, v2 = st.columns(2)
    with v1:
        risk_counts = filtered["risk_level"].astype(str).value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        st.write("Risk level distribution")
        st.bar_chart(risk_counts)

    with v2:
        if INDUSTRY_COL in filtered.columns:
            by_dept = filtered.groupby(INDUSTRY_COL, dropna=False)["risk_score"].mean().sort_values(ascending=False)
            st.write("Average risk score by department")
            st.bar_chart(by_dept)

    st.subheader("At-Risk Employees")
    high_risk = filtered[filtered["risk_level"].astype(str) == "High"].sort_values("risk_score", ascending=False)
    show_cols = [c for c in [INDUSTRY_COL, "profession", "age", TENURE_COL, "risk_score", "risk_level"] if c in high_risk.columns]
    st.dataframe(high_risk[show_cols], use_container_width=True)

    st.subheader("Actionable Retention Recommendations")
    if len(high_risk) > 0:
        selected_idx = st.selectbox("Pick an at-risk employee row", options=high_risk.index.tolist())
        employee = high_risk.loc[selected_idx]
        recs = retention_recommendations(employee, scored_df)
        for rec in recs:
            st.write(f"- {rec}")
    else:
        st.write("No high-risk employees in current filter.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Manual Employee Prediction")
    with st.sidebar.form("manual_form"):
        input_data = {}
        for col in model_bundle["feature_cols"]:
            if col in raw_df.select_dtypes(include=[np.number]).columns:
                median_val = float(pd.to_numeric(raw_df[col], errors="coerce").median())
                input_data[col] = st.number_input(col, value=median_val)
            else:
                choices = raw_df[col].dropna().astype(str).unique().tolist()
                default_choice = choices[0] if choices else ""
                input_data[col] = st.selectbox(col, options=choices if choices else [""], index=0)
        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        manual_df = pd.DataFrame([input_data])
        risk = float(best_model.predict_proba(manual_df)[0, 1])
        if risk < 0.4:
            level = "Low"
        elif risk < 0.7:
            level = "Medium"
        else:
            level = "High"

        st.sidebar.success(f"Predicted turnover risk: {risk:.2%} ({level})")
        manual_recs = retention_recommendations(manual_df.iloc[0], scored_df)
        st.sidebar.write("Recommended actions:")
        for rec in manual_recs:
            st.sidebar.write(f"- {rec}")


if __name__ == "__main__":
    main()
