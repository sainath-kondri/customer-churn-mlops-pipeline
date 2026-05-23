import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_processed_data(file_path):
    """Load processed churn dataset."""
    return pd.read_csv(file_path)


def encode_categorical_columns(df):
    """Encode categorical columns."""
    df = df.copy()

    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])

    return df


def prepare_train_test_data(df):
    """Prepare train and test datasets."""
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, y_train):
    """Train Logistic Regression and Random Forest models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_models(models, X_test, y_test):
    """Evaluate trained models."""
    best_model = None
    best_score = 0
    best_model_name = ""

    for name, model in models.items():
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nModel: {name}")
        print("Accuracy:", accuracy)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_model_name = name

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {best_score}")

    return best_model, best_model_name, best_score


def save_model(model, scaler, model_path):
    """Save trained model and scaler."""
    model_data = {
        "model": model,
        "scaler": scaler
    }

    with open(model_path, "wb") as file:
        pickle.dump(model_data, file)


if __name__ == "__main__":

    processed_data_path = "data/processed/processed_churn_data.csv"
    model_path = "models/churn_model.pkl"

    df = load_processed_data(processed_data_path)

    encoded_df = encode_categorical_columns(df)

    X_train, X_test, y_train, y_test, scaler = prepare_train_test_data(encoded_df)

    trained_models = train_models(X_train, y_train)

    best_model, best_model_name, best_score = evaluate_models(
        trained_models,
        X_test,
        y_test
    )

    save_model(best_model, scaler, model_path)

    print(f"\nSaved best model '{best_model_name}' to {model_path}")