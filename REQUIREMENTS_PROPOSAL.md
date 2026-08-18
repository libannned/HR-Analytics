# HR Analytics Turnover Predictor - Requirements Document (Proposal)

## 1. Purpose
This document defines the proposed requirements for an HR Analytics system that estimates employee turnover risk using historical employee data.

## 2. Scope
The proposed system will:
- Ingest employee data from CSV/Excel sources.
- Predict turnover risk probability for each employee.
- Classify risk levels (Low, Medium, High).
- Provide interactive analytics and filtering.
- Suggest actionable retention recommendations.

## 3. Inputs and Data Assumptions
Primary dataset: `turnover.csv`.
Expected fields include:
- Outcome: `event` (1 = leave, 0 = stay)
- Context: `industry`, `profession`, `stag`, `coach`, `traffic`, `way`, `greywage`
- Demographic: `age`, `gender`, `head_gender`
- Behavioral: `extraversion`, `independ`, `selfcontrol`, `anxiety`, `novator`

## 4. Functional Requirements
1. The system shall import employee data from CSV and Excel files.
2. The system shall support manual entry of employee attributes for one-off prediction.
3. The system shall preprocess numerical and categorical fields for model training.
4. The system shall train and compare at least two ML models.
5. The system shall compute turnover probability for each employee.
6. The system shall map probabilities into risk levels: Low, Medium, High.
7. The system shall provide filters by department, tenure, and risk level.
8. The system shall display risk charts and at-risk employee tables.
9. The system shall generate retention recommendations based on risk signals.
10. The system shall show an interpretation guide for risk scores.

## 5. Non-Functional Requirements
1. Usability: Interface should be understandable by HR users without ML background.
2. Performance: Typical datasets (about 1k rows) should process within a few seconds.
3. Reliability: System should handle missing values and file-encoding variation.
4. Maintainability: Components should be modular (ingestion, training, scoring, UI).
5. Portability: Solution should run locally and in cloud deployment.
6. Security/Privacy: Employee data handling should follow least-access principles.
7. Explainability: Users should be able to interpret risk levels and suggested actions.

## 6. Proposed Risk Interpretation Rules
- Low risk: score < 0.40
- Medium risk: 0.40 to 0.69
- High risk: score >= 0.70

## 7. Constraints
- Data quality and label quality affect prediction reliability.
- Encoded categorical values may require data dictionary clarification.
- Predictions should support, not replace, HR decision-making.

## 8. Acceptance Criteria (Proposal Stage)
1. Stakeholders approve this requirements document.
2. A prototype architecture (UML + workflow) is accepted.
3. Dataset schema and definitions are confirmed.
4. Model evaluation criteria (e.g., ROC-AUC) are agreed upon.
5. Deployment path and governance approach are approved.
