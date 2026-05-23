import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_processed_data(file_path):
    """Load processed churn dataset."""
    return pd.read_csv(file_path)


def encode_categorical_columns(df):
    """Apply one-hot encoding to categorical columns."""
    df = df.copy()

    encoded_df = pd.get_dummies(df, drop_first=True)

    return encoded_df


def split_features_target(df, target_column="Churn"):
    """Split dataframe into features and target."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return X, y


def split_train_test(X, y):
    """Split data into train and test sets using stratified sampling."""
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":

    processed_data_path = "data/processed/processed_churn_data.csv"

    df = load_processed_data(processed_data_path)

    print("\nProcessed Dataset Loaded Successfully")
    print("Original Shape:", df.shape)

    encoded_df = encode_categorical_columns(df)

    print("\nOne-Hot Encoding Completed")
    print("Encoded Shape:", encoded_df.shape)

    X, y = split_features_target(encoded_df)

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("\nFeature Engineering Completed Successfully")
    print("Training Features Shape:", X_train_scaled.shape)
    print("Testing Features Shape:", X_test_scaled.shape)
    print("Training Target Shape:", y_train.shape)
    print("Testing Target Shape:", y_test.shape)
    print("Total Features:", len(feature_names))

    print("\nClass Distribution in Full Dataset:")
    print(y.value_counts(normalize=True))

    print("\nClass Distribution in Training Set:")
    print(y_train.value_counts(normalize=True))

    print("\nClass Distribution in Testing Set:")
    print(y_test.value_counts(normalize=True))