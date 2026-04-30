import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def detect_column_types(df, target_col):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    return numeric_cols, categorical_cols


def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def encode_categoricals(df):
    binary_cols = [col for col in df.select_dtypes(include=["object"]).columns
                   if df[col].nunique() == 2]
    multi_cols = [col for col in df.select_dtypes(include=["object"]).columns
                  if df[col].nunique() > 2]

    encoders = {}

    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    return df, encoders


def remove_outliers_iqr(df, target_col):
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df[target_col] >= lower) & (df[target_col] <= upper)]
    logger.info(f"Outlier removal: {before - len(df)} rows dropped from target")
    return df


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(filepath, target_col="price", test_size=0.2):
    df = load_dataset(filepath)
    df = handle_missing_values(df)
    df = remove_outliers_iqr(df, target_col)
    df, encoders = encode_categoricals(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_names,
        "df_clean": df,
        "df_original": load_dataset(filepath),
    }
