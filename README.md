# Customer Churn MLOps Pipeline

An end-to-end Machine Learning and MLOps project for predicting customer churn using the Telco Customer Churn dataset.

## Live Demo

* Live API: https://customer-churn-api-7kb5.onrender.com
* Swagger Documentation: https://customer-churn-api-7kb5.onrender.com/docs

## Project Overview

This project demonstrates the complete machine learning lifecycle, starting from data preprocessing and feature engineering to model training, API development, containerization, and cloud deployment.

The objective is to predict whether a customer is likely to churn based on demographic and service-related information.

## Tech Stack

* Python
* Pandas
* Scikit-learn
* FastAPI
* Pydantic
* Docker
* Render
* Git & GitHub

## Project Workflow

Raw Data
→ Data Preprocessing
→ Feature Engineering
→ Model Training
→ Model Evaluation
→ Model Artifact Creation
→ FastAPI Prediction API
→ Docker Containerization
→ Cloud Deployment (Render)

## Features

* Data Cleaning and Preprocessing
* One-Hot Encoding
* Feature Scaling
* Logistic Regression
* Random Forest Classification
* Model Evaluation Metrics
* REST API using FastAPI
* Interactive Swagger UI
* Dockerized Application
* Cloud Deployment on Render

## API Endpoints

### Home Endpoint

GET /

Returns API status and model information.

### Prediction Endpoint

POST /predict

Accepts customer information and returns:

* Churn Prediction
* Churn Probability
* Model Used

## Deployment

The application is containerized using Docker and deployed on Render.

Public URL:

https://customer-churn-api-7kb5.onrender.com

## Future Improvements

* MLflow Experiment Tracking
* Monitoring and Logging
* CI/CD with GitHub Actions
* Automated Model Retraining
* Kubernetes Deployment

## Author

Sainath Kondri
