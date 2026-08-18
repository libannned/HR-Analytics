# Employee Turnover Predictor - Final Talk Track

## Slide 1 - Title
Today we present our final Employee Turnover Predictor, which combines risk analytics with actionable HR recommendations.

## Slide 2 - Problem and Goal
Turnover is expensive and often reactive. Our goal is to identify likely turnover earlier and give managers practical intervention guidance.

## Slide 3 - System Overview
The system takes employee factors from turnover.csv, predicts turnover risk, and presents results through interactive filtering and scenario analysis.

## Slide 4 - Requirements Summary
Functionally, we import data, score risk, filter insights, and produce recommendations. Non-functionally, we focus on readability, speed, modularity, and responsible data handling.

## Slide 5 - Data and Features
The dataset has 1,129 records and 16 fields. The target is event, where 1 means left and 0 means stayed. Features combine contextual, demographic, and behavioral signals.

## Slide 6 - Class Diagram
The class model is layered into UI, service, and data classes. This keeps each class focused and makes changes easier to isolate.

## Slide 7 - Behavioral Diagram
This sequence diagram captures runtime behavior: app initialization, data loading, scoring, rendering, and user-driven prediction flows.

## Slide 8 - Machine Learning Pipeline
We preprocess data, split train/test, train two models, and choose the best based on ROC-AUC before scoring employees.

## Slide 9 - Risk Scoring and Decision Support
Risk is mapped into Low, Medium, and High thresholds. High-risk cases trigger focused retention recommendations for managers.

## Slide 10 - Implementation Status and Demo Plan
We have a working Streamlit implementation and a React UI prototype. In the demo: load data, filter, inspect high-risk employees, then run a scenario prediction.

## Slide 11 - Limitations and Next Steps
Current limits are data quality and model explainability depth. Next steps include stronger testing, CI/CD, fairness checks, and tracking intervention outcomes.

## Slide 12 - Questions
Thank you. We welcome questions on architecture, model decisions, and future improvements.
