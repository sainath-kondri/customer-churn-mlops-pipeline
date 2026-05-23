import json
import pickle
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.feature_engineering import (
    load_processed_data,
    encode_categorical_columns,
    split_features_target,
    split_train_test,
    scale_features,
)


def train_models(X_train, y_train):
    """Train multiple classification models."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=42
        )
    }

    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    return trained_models


def evaluate_models(models, X_test, y_test):
    """Evaluate models and select the best one using F1-score."""

    results = {}
    best_model = None
    best_model_name = None
    best_f1 = 0

    for model_name, model in models.items():

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "classification_report": classification_report(
                y_test,
                y_pred,
                output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = model_name

    return best_model, best_model_name, results


def save_model_artifact(model, scaler, feature_names, model_name, metrics, artifact_path):
    """Save trained model, scaler, feature names, and metadata."""

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": model_name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics[model_name]
    }

    with open(artifact_path, "wb") as file:
        pickle.dump(artifact, file)


def save_metrics_report(metrics, report_path):
    """Save model evaluation metrics as JSON."""

    with open(report_path, "w") as file:
        json.dump(metrics, file, indent=4)


if __name__ == "__main__":

    processed_data_path = "data/processed/processed_churn_data.csv"
    model_artifact_path = "models/churn_model.pkl"
    metrics_report_path = "models/model_metrics.json"

    Path("models").mkdir(parents=True, exist_ok=True)

    df = load_processed_data(processed_data_path)

    encoded_df = encode_categorical_columns(df)

    X, y = split_features_target(encoded_df)

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    trained_models = train_models(X_train_scaled, y_train)

    best_model, best_model_name, metrics = evaluate_models(
        trained_models,
        X_test_scaled,
        y_test
    )

    save_model_artifact(
        model=best_model,
        scaler=scaler,
        feature_names=feature_names,
        model_name=best_model_name,
        metrics=metrics,
        artifact_path=model_artifact_path
    )

    save_metrics_report(metrics, metrics_report_path)

    print(f"\nBest Model Selected: {best_model_name}")
    print(f"Model artifact saved to: {model_artifact_path}")
    print(f"Metrics report saved to: {metrics_report_path}")