from enum import Enum
import mlflow
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from pipeline import (
    MissingStrategy,
    EducationMode,
    FairnessMode,
    training_pipeline,
    evaluation_pipeline,
)

def ml_flow_train(adult_df: pd.DataFrame, config: dict):
    mlflow.set_tracking_uri("http://host.docker.internal:5000") # <-- This line is if we are using the dev container
    # mlflow.set_tracking_uri("http://localhost:5000") # <-- This line is if we are running the script locally
    mlflow.set_experiment("Default")
    # Enable autologging for scikit-learn
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        result, y_true = training_pipeline(adult_df, config=config)

        results = evaluation_pipeline(y_true, result)

        # log parameters
        for key, value in config.items():
            if isinstance(value, type) and issubclass(value, BaseEstimator):
                mlflow.log_param(key, value.__name__)
            else:
                mlflow.log_param(key, value if not isinstance(value, Enum) else value.value)

        # Log evaluation metrics
        for metric, value in results.items():
            mlflow.log_metric(metric, value)

if __name__ == "__main__":
    adult_df = pd.read_csv("data/adult.csv")
    config = {
        "missing_strategy": MissingStrategy.UNKNOWN,
        "education_mode": EducationMode.NUM,
        "fairness_mode": FairnessMode.NONE,
        "target_col": "income",
        "sensitive_cols": ["sex", "race"],
        "model_type": GradientBoostingClassifier,  # sklearn classifier class, e.g. LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
    }

    ml_flow_train(adult_df, config=config)