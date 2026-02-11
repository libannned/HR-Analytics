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

FIELD_LABELS = {
    "event": "Turnover Outcome",
    "stag": "Tenure",
    "gender": "Gender",
    "age": "Age",
    "industry": "Department",
    "profession": "Job Function",
    "traffic": "Recruitment Source",
    "coach": "Coaching Support",
    "head_gender": "Manager Gender",
    "greywage": "Compensation Type",
    "way": "Commute Method",
    "extraversion": "Extraversion",
    "independ": "Independence",
    "selfcontrol": "Self-Control",
    "anxiety": "Anxiety",
    "novator": "Innovation Openness",
    "risk_score": "Risk Score",
    "risk_level": "Risk Level",
}

VALUE_LABELS = {
    "gender": {"f": "Female", "m": "Male"},
    "head_gender": {"f": "Female", "m": "Male"},
    "coach": {"yes": "Has coach", "no": "No coach", "my head": "Direct manager"},
    "greywage": {"white": "Formal/official pay", "grey": "Informal/gray pay"},
    "way": {"bus": "Bus", "car": "Car", "foot": "Walk"},
    "traffic": {
        "youjs": "Online job board",
        "empjs": "Employer career site",
        "rabrecNErab": "Recruiter/network channel",
        "recNErab": "Recruiter channel",
        "referal": "Employee referral",
        "friends": "Friend/network referral",
        "advert": "Advertisement",
        "KA": "Campus/agency channel",
    },
}


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


def pretty_label(col_name):
    return FIELD_LABELS.get(col_name, col_name.replace("_", " ").title())


def pretty_value(col_name, value):
    mapped = VALUE_LABELS.get(col_name, {}).get(str(value), value)
    return mapped


def build_choice_map(col_name, raw_values):
    choice_map = {}
    for raw in raw_values:
        display = str(pretty_value(col_name, raw))
        if display in choice_map and choice_map[display] != raw:
            display = f"{display} ({raw})"
        choice_map[display] = raw
    return choice_map


def to_display_dataframe(df):
    display_df = df.copy()
    for col in display_df.columns:
        if col in VALUE_LABELS:
            display_df[col] = display_df[col].map(
                lambda v: pretty_value(col, v) if pd.notna(v) else v
            )
    renamed = {c: pretty_label(c) for c in display_df.columns}
    return display_df.rename(columns=renamed)


def inject_custom_styles():
    st.markdown(
        """
        <style>
        .app-subtitle {
            color: #475569;
            font-size: 1rem;
            margin-bottom: 0.8rem;
        }
        .section-card {
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }
        .risk-chip {
            display: inline-block;
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid transparent;
        }
        .risk-chip.low {
            background: #ecfdf3;
            color: #065f46;
            border-color: #a7f3d0;
        }
        .risk-chip.medium {
            background: #fffbeb;
            color: #92400e;
            border-color: #fde68a;
        }
        .risk-chip.high {
            background: #fef2f2;
            color: #991b1b;
            border-color: #fecaca;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_chip(level):
    level = str(level).lower()
    css = "low" if level == "low" else "medium" if level == "medium" else "high"
    text = "Low" if css == "low" else "Medium" if css == "medium" else "High"
    return f'<span class="risk-chip {css}">{text} Risk</span>'


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
- High Anxiety: potential stress/support issue.
- Low Self-Control: may need clearer goals and structured planning.
- Short Tenure: onboarding/early-tenure risk.
- No Coaching Support: add mentor/coach support.
- Informal/Gray Pay: possible compensation trust/fairness issue.

**Important Note**
- This model is an early-warning signal, not proof. Confirm with manager context and direct employee conversation.
"""
        )

    with st.expander("Data Dictionary (Plain English)", expanded=False):
        dictionary_rows = [
            ("Tenure", "Time employee has been with the organization."),
            ("Recruitment Source", "Where employee was hired from (referral, job board, etc.)."),
            ("Coaching Support", "Whether the employee has coaching/manager support."),
            ("Compensation Type", "Compensation arrangement category."),
            ("Extraversion", "How outgoing/social the employee tends to be."),
            ("Independence", "How self-directed the employee tends to be."),
            ("Self-Control", "How consistently the employee self-manages tasks/behavior."),
            ("Anxiety", "Stress or tension tendency level."),
            ("Innovation Openness", "Willingness to try new approaches."),
        ]
        st.table(pd.DataFrame(dictionary_rows, columns=["Factor", "Meaning"]))


def main():
    st.set_page_config(page_title="HR Analytics: Turnover Predictor", layout="wide")
    inject_custom_styles()
    st.title("HR Analytics: Employee Turnover Predictor")
    st.markdown(
        '<div class="app-subtitle">Estimate turnover risk, identify employees who may need support, and suggest practical retention actions.</div>',
        unsafe_allow_html=True,
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
    model_score_display = score_df.rename(
        columns={
            "model": "Model",
            "roc_auc": "ROC-AUC",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1 Score",
        }
    )
    st.dataframe(
        model_score_display.style.format(
            {c: "{:.3f}" for c in model_score_display.columns if c != "Model"}
        ),
        use_container_width=True,
    )
    st.info(f"Best model by ROC-AUC: {best_name}")

    st.subheader("Risk Score Dashboard")
    render_risk_interpretation_guide()

    display_df = to_display_dataframe(scored_df)
    industry_label = pretty_label(INDUSTRY_COL)
    tenure_label = pretty_label(TENURE_COL)
    risk_level_label = pretty_label("risk_level")
    risk_score_label = pretty_label("risk_score")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dashboard Filters")
    departments = (
        sorted(display_df[industry_label].dropna().astype(str).unique())
        if industry_label in display_df.columns
        else []
    )
    tenure_min = float(scored_df[TENURE_COL].min()) if TENURE_COL in scored_df.columns else 0.0
    tenure_max = float(scored_df[TENURE_COL].max()) if TENURE_COL in scored_df.columns else 0.0

    with st.sidebar:
        selected_departments = st.multiselect(
            "Department", departments, default=departments
        )
        selected_risk = st.multiselect(
            "Risk Level",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
        )
        selected_tenure = st.slider(
            "Tenure Range",
            min_value=tenure_min,
            max_value=tenure_max,
            value=(tenure_min, tenure_max),
        )

    filtered = scored_df.copy()
    filtered_display = display_df.copy()

    if selected_departments and industry_label in filtered_display.columns:
        selected_idx = filtered_display[
            filtered_display[industry_label].astype(str).isin(selected_departments)
        ].index
        filtered = filtered.loc[selected_idx]
        filtered_display = filtered_display.loc[selected_idx]
    if selected_risk:
        selected_idx = filtered_display[
            filtered_display[risk_level_label].astype(str).isin(selected_risk)
        ].index
        filtered = filtered.loc[selected_idx]
        filtered_display = filtered_display.loc[selected_idx]
    if TENURE_COL in filtered.columns:
        filtered = filtered[
            (filtered[TENURE_COL] >= selected_tenure[0]) & (filtered[TENURE_COL] <= selected_tenure[1])
        ]
        filtered_display = filtered_display.loc[filtered.index]

    tabs = st.tabs(["Overview", "At-Risk Employees", "Manual Prediction", "Guide"])

    with tabs[0]:
        st.markdown(
            '<div class="section-card"><strong>Dashboard Summary</strong><br/>Use sidebar filters to narrow to any team, tenure band, or risk tier.</div>',
            unsafe_allow_html=True,
        )
        k1, k2, k3 = st.columns(3)
        k1.metric("Employees (Filtered)", f"{len(filtered):,}")
        k2.metric(
            "At-Risk (High)",
            f"{(filtered_display[risk_level_label].astype(str) == 'High').sum():,}",
        )
        k3.metric(
            "Avg. Risk Score",
            f"{filtered_display[risk_score_label].mean():.2f}" if len(filtered_display) else "N/A",
        )

        v1, v2 = st.columns(2)
        with v1:
            risk_counts = (
                filtered_display[risk_level_label]
                .astype(str)
                .value_counts()
                .reindex(["Low", "Medium", "High"], fill_value=0)
            )
            st.write("Risk level distribution")
            st.bar_chart(risk_counts)
        with v2:
            if industry_label in filtered_display.columns:
                by_dept = (
                    filtered_display.groupby(industry_label, dropna=False)[risk_score_label]
                    .mean()
                    .sort_values(ascending=False)
                )
                st.write("Average risk score by department")
                st.bar_chart(by_dept)

        st.subheader("Model Performance")
        st.dataframe(
            model_score_display.style.format(
                {c: "{:.3f}" for c in model_score_display.columns if c != "Model"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.info(f"Best model by ROC-AUC: {best_name}")

    with tabs[1]:
        high_risk = filtered_display[
            filtered_display[risk_level_label].astype(str) == "High"
        ].sort_values(risk_score_label, ascending=False)
        high_risk_raw = filtered.loc[high_risk.index]
        show_cols = [
            c
            for c in [
                industry_label,
                pretty_label("profession"),
                pretty_label("age"),
                tenure_label,
                risk_score_label,
                risk_level_label,
            ]
            if c in high_risk.columns
        ]
        st.subheader("High-Risk Employee List")
        st.dataframe(high_risk[show_cols], use_container_width=True, hide_index=True)

        st.subheader("Actionable Retention Recommendations")
        if len(high_risk_raw) > 0:
            option_map = {}
            for idx in high_risk_raw.index.tolist():
                dept = str(high_risk.loc[idx, industry_label]) if industry_label in high_risk.columns else "N/A"
                score = float(high_risk.loc[idx, risk_score_label])
                label = f"Employee {idx} | {dept} | score {score:.2f}"
                option_map[label] = idx
            selected_label = st.selectbox("Select employee", options=list(option_map.keys()))
            selected_idx = option_map[selected_label]
            employee = high_risk_raw.loc[selected_idx]
            level = high_risk.loc[selected_idx, risk_level_label]
            st.markdown(
                f"Selected risk level: {risk_chip(level)}",
                unsafe_allow_html=True,
            )
            recs = retention_recommendations(employee, scored_df)
            for rec in recs:
                st.write(f"- {rec}")
        else:
            st.write("No high-risk employees in the current filter.")

    with tabs[2]:
        st.subheader("Manual Employee Prediction")
        st.caption("Use this form for a hypothetical or new employee scenario.")
        with st.form("manual_form"):
            input_data = {}
            for col in model_bundle["feature_cols"]:
                if col in raw_df.select_dtypes(include=[np.number]).columns:
                    median_val = float(pd.to_numeric(raw_df[col], errors="coerce").median())
                    input_data[col] = st.number_input(pretty_label(col), value=median_val)
                else:
                    choices = sorted(raw_df[col].dropna().astype(str).unique().tolist())
                    choice_map = build_choice_map(col, choices)
                    options = list(choice_map.keys()) if choice_map else [""]
                    selected_display = st.selectbox(pretty_label(col), options=options, index=0)
                    input_data[col] = choice_map.get(selected_display, "")
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

            st.markdown(f"Prediction: {risk_chip(level)}", unsafe_allow_html=True)
            st.metric("Predicted Turnover Probability", f"{risk:.2%}")
            manual_recs = retention_recommendations(manual_df.iloc[0], scored_df)
            st.write("Recommended actions:")
            for rec in manual_recs:
                st.write(f"- {rec}")

    with tabs[3]:
        render_risk_interpretation_guide()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Manual Prediction")
    with st.sidebar.form("manual_form_sidebar"):
        input_data = {}
        for col in model_bundle["feature_cols"]:
            if col in raw_df.select_dtypes(include=[np.number]).columns:
                median_val = float(pd.to_numeric(raw_df[col], errors="coerce").median())
                input_data[col] = st.number_input(pretty_label(col), value=median_val)
            else:
                choices = sorted(raw_df[col].dropna().astype(str).unique().tolist())
                choice_map = build_choice_map(col, choices)
                options = list(choice_map.keys()) if choice_map else [""]
                selected_display = st.selectbox(pretty_label(col), options=options, index=0)
                input_data[col] = choice_map.get(selected_display, "")
        submitted = st.form_submit_button("Predict in Sidebar")

    if submitted:
        manual_df = pd.DataFrame([input_data])
        risk = float(best_model.predict_proba(manual_df)[0, 1])
        if risk < 0.4:
            level = "Low"
        elif risk < 0.7:
            level = "Medium"
        else:
            level = "High"

        st.sidebar.markdown(f"Result: {risk_chip(level)}", unsafe_allow_html=True)
        st.sidebar.metric("Predicted Probability", f"{risk:.2%}")
        manual_recs = retention_recommendations(manual_df.iloc[0], scored_df)
        st.sidebar.write("Recommended actions:")
        for rec in manual_recs:
            st.sidebar.write(f"- {rec}")


if __name__ == "__main__":
    main()
