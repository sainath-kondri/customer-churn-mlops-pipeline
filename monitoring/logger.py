import csv
import os
from datetime import datetime


LOG_FILE_PATH = "monitoring/prediction_logs.csv"


def log_prediction(input_data, prediction, probability, model_name):
    """Log prediction request and response."""

    os.makedirs("monitoring", exist_ok=True)

    file_exists = os.path.isfile(LOG_FILE_PATH)

    with open(LOG_FILE_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "model_name",
                "prediction",
                "churn_probability",
                "input_data"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            prediction,
            probability,
            input_data
        ])