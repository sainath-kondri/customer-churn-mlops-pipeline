import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_processed_data(file_path):
    """Load processed churn dataset."""

    df = pd.read_csv(file_path)

    return df


def encode_categorical_columns(df):
    """Encode categorical columns using Label Encoding."""

    df = df.copy()

    categorical_columns = df.select_dtypes(include=["object"]).columns

    label_encoders = {}

    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    return df, label_encoders


def split_features_target(df):
    """Split dataset into features and target."""

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y


def split_train_test(X, y):
    """Split data into training and testing sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Scale numerical features."""

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":

    processed_data_path = "data/processed/processed_churn_data.csv"

    df = load_processed_data(processed_data_path)

    print("\nProcessed Dataset Loaded:")
    print(df.head())

    encoded_df, label_encoders = encode_categorical_columns(df)

    print("\nEncoded Dataset:")
    print(encoded_df.head())

    X, y = split_features_target(encoded_df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("\nFeature Engineering Completed Successfully")
    print("X_train shape:", X_train_scaled.shape)
    print("X_test shape:", X_test_scaled.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)