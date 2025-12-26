# Breast Cancer Detection System

## Overview

This project implements an end-to-end machine learning system for breast
cancer classification. The application predicts whether a breast tumor
is **Benign** or **Malignant** using two supervised learning algorithms:
**Decision Tree** and **Logistic Regression**.

The system includes data preprocessing, model training and evaluation,
model serialization, backend integration using FastAPI, and a deployed
web-based frontend for user interaction.

------------------------------------------------------------------------

## Dataset

The dataset used in this project is sourced from **Kaggle** and is based
on the Breast Cancer Wisconsin Diagnostic Dataset.

-   **Format:** CSV\
-   **Target variable:** `diagnosis`
    -   `0` --- Benign\
    -   `1` --- Malignant

A subset of numerical features was selected for model training.

------------------------------------------------------------------------

## Data Preprocessing

The following preprocessing steps were applied:

-   Initial data inspection (`head`, `info`, `describe`)
-   Verification of missing values
-   Removal of duplicate records
-   Outlier removal using the **Z-score method**
-   Feature selection of relevant numerical attributes
-   Train--validation split (80% training, 20% validation)
-   Feature scaling using `StandardScaler` within Scikit-learn pipelines

------------------------------------------------------------------------

## Machine Learning Models

### Decision Tree Classifier

-   Implemented using Scikit-learn
-   Trained within a pipeline that includes feature scaling

### Logistic Regression

-   Implemented using Scikit-learn
-   Configured with an increased maximum number of iterations to ensure
    convergence
-   Trained within a pipeline that includes feature scaling

------------------------------------------------------------------------

## Model Evaluation

Model performance was evaluated using accuracy scores and classification
reports.

------------------------------------------------------------------------

## Model Serialization

Trained models were saved using **joblib**:

-   `dt_model.pkl`
-   `lr_model.pkl`

------------------------------------------------------------------------

## Backend Implementation

The backend is developed using **FastAPI** and provides a RESTful API
for inference.

### API Endpoint

POST `/predict`

------------------------------------------------------------------------

## Frontend

The frontend is built using HTML, Tailwind CSS, and JavaScript to
provide a simple user interface for predictions.

------------------------------------------------------------------------

## Deployment

-   Backend deployed on Render\
-   Frontend deployed on Vercel

Live application:\
https://breast-cancer-detector-th46.vercel.app/

------------------------------------------------------------------------

## Project Structure

Breast_cancer_detector/
│
├── notebook/
│   └── breast_cancer_pipeline.ipynb
│
├── backend/
│   ├── main.py
│   └── models/
│       ├── dt_model.pkl
│       └── lr_model.pkl
│
├── data/
│   └── breast_cancer_data.csv
│
├── index.html
└── README.md


------------------------------------------------------------------------

## Author

Dagmawit Sisay(UGR/2038/15)
Addis Ababa University
