import pickle
import pandas as pd

from fastapi import FastAPI
from api.schema import CustomerData
from monitoring.logger import log_prediction


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using a trained machine learning model.",
    version="1.0.0"
)


MODEL_PATH = "models/churn_model.pkl"


with open(MODEL_PATH, "rb") as file:
    model_artifact = pickle.load(file)


model = model_artifact["model"]
scaler = model_artifact["scaler"]
feature_names = model_artifact["feature_names"]
model_name = model_artifact["model_name"]


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "model": model_name
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": model_name
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    input_data = pd.DataFrame([customer.model_dump()])

    input_encoded = pd.get_dummies(input_data)

    input_encoded = input_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )

    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"
    churn_probability = round(float(probability), 4)

    log_prediction(
        input_data=customer.model_dump(),
        prediction=result,
        probability=churn_probability,
        model_name=model_name
    )

    return {
        "prediction": result,
        "churn_probability": churn_probability,
        "model_used": model_name
    }