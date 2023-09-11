# Loan-Eligibility-Classification-and-Drift-Monitoring-Pipeline-with-Dockerized-Airflow

# Overview

The Machine Learning Model Project is designed to monitor the performance of machine learning models over time, detect data drifts, and trigger retraining when necessary. This project provides a comprehensive framework to ensure that machine learning models continue to deliver accurate predictions as new data becomes available.

# Features

Data Quality Check: The project includes data quality checks to identify missing values, duplicates, and label conflicts in the dataset. If data quality issues are detected, training can be skipped until the issues are resolved.

Data Drift Detection: The project monitors data drift between reference and current datasets. It checks for changes in data distributions, feature-label correlations, and outliers. If significant drift is detected, it triggers retraining.

Model Drift Detection: Model drift is assessed by comparing the predictions of a pre-trained model on reference and current datasets. If the model's performance degrades significantly, it indicates the need for retraining.

Automatic Data Preprocessing: The project automatically preprocesses the data for training and inference. Categorical variables are encoded, missing values are imputed, and numerical variables are scaled as needed.

Data Collection: Gather reference and current datasets. Define the target variable and predictor variables.

Run Data Quality Check:

Execute the skip_train function to check if data quality issues exist that should prevent training.
Review the output to determine if training should be skipped due to data quality.

Run Data Drift Check:

Use the check_data_drift function to assess data drift between reference and current datasets.
Analyze the results to decide if retraining is required based on data drift.

Data Preprocessing:

Preprocess both reference and current datasets using the preprocess.preprocess_data function.
Ensure that the data is in the appropriate format for model training and inference.
Model Drift Check:

Load the pre-trained model.
Execute the check_model_drift function to assess model drift between reference and current datasets.
Determine if retraining is necessary based on model drift analysis.

Retraining:

If data drift or model drift indicates the need for retraining, train a new model using the reference dataset.
Save the new model for future predictions.
