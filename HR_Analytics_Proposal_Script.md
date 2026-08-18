# HR Analytics Proposal Script (Future-Tense)

## Slide 1 - Title
Today I am presenting a proposal for an HR Analytics Employee Turnover Predictor. This is a concept-stage design based on the requirement PDF and the turnover dataset.

## Slide 2 - Problem Statement
Employee turnover is costly and often detected too late. Teams usually react after resignation signals become obvious. This proposal focuses on early risk detection and proactive action.

## Slide 3 - Project Vision
The idea is to design a dashboard that can estimate turnover risk and guide managers toward targeted retention actions. The intended outcome is better prioritization and faster intervention.

## Slide 4 - Available Inputs
At this stage, we only rely on two assets: a requirement PDF that defines expected features, and turnover.csv containing employee factors and turnover labels.

## Slide 5 - Functional Requirements
The proposed system should import HR data, run prediction models, classify risk levels, provide filtering, and offer actionable recommendations. It should also allow manual employee input for one-off predictions.

## Slide 6 - Non-Functional Requirements
The system should be easy to use, perform quickly on standard datasets, and remain reliable with imperfect data. It should also be maintainable and protect employee data privacy.

## Slide 7 - Proposed UML
This UML view outlines the software structure. The UI layer will coordinate data loading, model training, risk scoring, and recommendation generation. Each component has a clear responsibility to keep the design modular.

## Slide 8 - Proposed Workflow
The flow will start with data ingestion, continue through preprocessing and model comparison, then produce risk scores and risk levels. The final step will be an interactive dashboard for filtering and decision support.

## Slide 9 - Proposed Data Factors
The model will use context, demographic, and behavioral factors from turnover.csv. The target variable is event, where 1 indicates leaving and 0 indicates staying.

## Slide 10 - Proposed Modeling Strategy
The initial baseline will compare Logistic Regression and Random Forest. ROC-AUC will guide model selection. Predicted probabilities will be grouped into low, medium, and high risk bands for practical interpretation.

## Slide 11 - Expected Value and Risks
The expected value is earlier intervention and more focused HR effort. The main risks are bias and data quality limitations. Therefore, this system should support decisions, not replace human judgment.

## Slide 12 - Next Steps
The next phase would formalize the data dictionary, implement a prototype, validate it with stakeholders, and prepare a deployment and governance plan.
