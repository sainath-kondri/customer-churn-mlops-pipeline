import pandas as pd
import os


def load_data(file_path):
    """Load customer churn dataset."""
    return pd.read_csv(file_path)


def clean_data(df):
    """Clean customer churn dataset."""

    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def save_processed_data(df, output_path):
    """Save cleaned dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":

    raw_data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    processed_data_path = "data/processed/processed_churn_data.csv"

    df = load_data(raw_data_path)

    print("\nBefore Cleaning:")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)

    cleaned_df = clean_data(df)

    print("\nAfter Cleaning:")
    print(cleaned_df.head())
    print("\nShape:", cleaned_df.shape)
    print("\nMissing Values:")
    print(cleaned_df.isnull().sum())
    print("\nData Types:")
    print(cleaned_df.dtypes)

    save_processed_data(cleaned_df, processed_data_path)

    print(f"\nProcessed data saved to: {processed_data_path}")