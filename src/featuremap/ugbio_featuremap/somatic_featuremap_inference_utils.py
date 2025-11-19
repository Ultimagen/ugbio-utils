import sys

import numpy as np
import pandas as pd
import xgboost
from sklearn.preprocessing import LabelEncoder
from ugbio_core.logger import logger


def load_model(xgb_model_file: str) -> "xgboost.XGBClassifier":
    """
    Load a pre-trained XGBoost model from a file.

    Args:
        xgb_model_file (str): Path to the XGBoost model file.

    Returns:
        xgboost.XGBClassifier: The loaded XGBoost classifier model.
    """
    # load xgb model
    xgb_clf_es = xgboost.XGBClassifier()
    xgb_clf_es.load_model(xgb_model_file)
    return xgb_clf_es


def set_categorical_columns(df):
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    le = LabelEncoder()
    for col in categorical_columns:
        df.loc[:, col] = le.fit_transform(df[col].astype(str))
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")


def predict(xgb_model: "xgboost.XGBClassifier", df_calls: "pd.DataFrame") -> "np.ndarray":
    """
    Generate prediction probabilities for the positive class using a trained XGBoost model.

    Args:
        xgb_model (xgboost.XGBClassifier): Trained XGBoost classifier with accessible feature names.
        df_calls (pd.DataFrame): DataFrame containing feature columns required by the model.

    Returns:
        np.ndarray: Array of predicted probabilities for the positive class ("1").
    """
    model_features = xgb_model.get_booster().feature_names
    missing_features = [f for f in model_features if f not in df_calls.columns]
    if len(missing_features) > 0:
        logger.error(f"The following model features are missing from input DataFrame: {missing_features}")
        sys.exit(1)
    X = df_calls[model_features]  # noqa: N806

    set_categorical_columns(X)
    # for col in X.select_dtypes(include="object").columns:
    #     X[col] = X[col].astype("category")

    probabilities = xgb_model.predict_proba(X)
    df_probabilities = pd.DataFrame(probabilities, columns=["0", "1"])
    return df_probabilities["1"].to_numpy()
