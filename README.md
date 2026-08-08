# Telco-Customer-Churn-Prediction
Predicting which telecom customers are likely to churn, and explaining why, so retention efforts can be targeted rather than blanket.

## Overview
The project uses machine learning techniques to try to predict which customers would churn, based on various features such as payment methods and how long they have been a customer.  The following approach was used:
- EDA and preprocessing: examined churn rate by demographic, service, and account features. We confirmed there was a 73/27 class imbalance and cleaned the data ready for analysis later on such as fixing data types and implementing one-hot encoding for categorical features
- Modelling: trained and compared three classifiers (Decision Tree, Random Forest, and XGBoost) using:
  - class weighting to handle the imbalance target
  - stratified 5-fold cross-validation on the training set to get a stable performance estimate before touching the holdout set
  - early stopping for the XGBoost model instead of a fixed tree count which may not be optimal
- Evaluation: precision, recall, F1, and ROG AUC
- Explainability: compared feature importances across XGBoost and Random Forest to make sure conclusions weren't an artifact of any single model's internal mechanics.

## Tech Stack
Python; pandas; scikit-learn; XGBoost; matplotlib; seaborn
