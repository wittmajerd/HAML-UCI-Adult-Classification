from collections import defaultdict
from enum import Enum
import logging

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio 


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class MissingStrategy(str, Enum):
    DROP = "DROP"
    UNKNOWN = "UNKNOWN"


class EducationMode(str, Enum):
    NUM = "NUM"
    CAT = "CAT"
    BOTH = "BOTH"


class FairnessMode(str, Enum):
    NONE = "NONE"
    DROP = "DROP"
    REWEIGH = "REWEIGH"
    MASK = "MASK"

class Model(str, Enum):
    LogReg = "LogReg"
    RF = "RF"
    GB = "GB"


def prepare_raw_df(
    df,
    missing_strategy: MissingStrategy = MissingStrategy.UNKNOWN,
    education_mode: EducationMode = EducationMode.NUM,
    target_col: str = "income",
):
    """Handle '?' values, choose education representation, and make target boolean."""
    if not isinstance(missing_strategy, MissingStrategy):
        raise TypeError("missing_strategy must be MissingStrategy enum")
    if not isinstance(education_mode, EducationMode):
        raise TypeError("education_mode must be EducationMode enum")

    df = df.copy()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    df = df.replace("?", pd.NA)

    if target_col in df.columns:
        df[target_col] = df[target_col].astype(str).str.replace(".", "", regex=False)
        df[target_col] = df[target_col] == ">50K"

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Use float for numeric features so MLflow signature remains compatible
    # with missing values that may appear at inference time. - if we use mlflow.sklearn.autolog() we get a warning for this
    # if numeric_cols:
    #     df[numeric_cols] = df[numeric_cols].astype("float64")

    if missing_strategy == MissingStrategy.DROP:
        df = df.dropna().reset_index(drop=True)
    else:
        cat_cols = [col for col in df.columns if col not in numeric_cols]
        for col in cat_cols:
            df[col] = df[col].fillna("Unknown")
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

    edu_num_col = "education_num" if "education_num" in df.columns else "education.num"
    edu_cat_col = "education"

    if education_mode == EducationMode.NUM and edu_cat_col in df.columns:
        df = df.drop(columns=[edu_cat_col])
    if education_mode == EducationMode.CAT and edu_num_col in df.columns:
        df = df.drop(columns=[edu_num_col])

    return df


# Kamiran és Calders, 2012
def _manual_reweighing(y, sensitive_df):
    """Reweighing: weight(group, label) = P(group) * P(label) / P(group, label)."""
    group_keys = sensitive_df.astype(str).agg("|".join, axis=1)
    n = len(y)

    group_counts = group_keys.value_counts().to_dict()
    label_counts = y.value_counts().to_dict()
    joint_counts = defaultdict(int)

    for g, label in zip(group_keys, y):
        joint_counts[(g, label)] += 1

    weights = []
    for g, label in zip(group_keys, y):
        p_g = group_counts[g] / n
        p_y = label_counts[label] / n
        p_gy = joint_counts[(g, label)] / n
        weights.append((p_g * p_y) / p_gy if p_gy > 0 else 1.0)

    return np.asarray(weights, dtype=float)


class MetaModel(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        base_model,
        fairness_mode: FairnessMode = FairnessMode.NONE,
        sensitive_cols=None,
    ):
        if not isinstance(fairness_mode, FairnessMode):
            raise TypeError("fairness_mode must be FairnessMode enum")

        self.base_model = base_model
        self.fairness_mode = fairness_mode
        self.sensitive_cols = sensitive_cols if sensitive_cols is not None else ["sex", "race"]
        self.feature_columns_ = None
        # One-hot encoding is always applied via pd.get_dummies.
        # Numeric scaling is applied only for LogisticRegression to keep behavior consistent across models.
        self.scale_numeric = isinstance(self.base_model, LogisticRegression)

    def _ensure_df(self, X):
        return X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def _ensure_series(self, y):
        if isinstance(y, pd.Series):
            return y.reset_index(drop=True)
        return pd.Series(y).reset_index(drop=True)

    def _apply_train_fairness(self, X_train, y_train, sample_weight):
        available_sensitive = [c for c in self.sensitive_cols if c in X_train.columns]

        if self.fairness_mode == FairnessMode.DROP and available_sensitive:
            X_train = X_train.drop(columns=available_sensitive)

        if self.fairness_mode == FairnessMode.REWEIGH:
            if sample_weight is None:
                if available_sensitive:
                    sample_weight = _manual_reweighing(y_train, X_train[available_sensitive])
                else:
                    sample_weight = np.ones(len(y_train), dtype=float)
            else:
                sample_weight = np.asarray(sample_weight, dtype=float)

        return X_train, y_train, sample_weight

    def _apply_inference_fairness(self, X):
        available_sensitive = [c for c in self.sensitive_cols if c in X.columns]
        if self.fairness_mode == FairnessMode.DROP and available_sensitive:
            X = X.drop(columns=available_sensitive)
        if self.fairness_mode == FairnessMode.MASK and available_sensitive:
            X = X.copy()
            for col in available_sensitive:
                X[col] = "Masked"
        return X

    def _encode_train(self, X_train):
        X_train = X_train.copy()
        if self.scale_numeric:
            numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
            self.numeric_cols_ = numeric_cols
            if numeric_cols:
                self.scaler = StandardScaler().fit(X_train[numeric_cols])
                self.numeric_means_ = pd.Series(self.scaler.mean_, index=numeric_cols)
                X_train[numeric_cols] = self.scaler.transform(X_train[numeric_cols])
            else:
                self.scaler = None
                self.numeric_means_ = None

        # Always one-hot encode categoricals.
        X_train_enc = pd.get_dummies(X_train, drop_first=False)
        self.feature_columns_ = X_train_enc.columns.tolist()
        return X_train_enc

    def _encode_inference(self, X):
        if self.feature_columns_ is None:
            raise RuntimeError("MetaModel must be fitted before inference")

        X = X.copy()

        if self.scale_numeric:
            if self.scaler is not None:
                numeric_cols = list(self.numeric_cols_)
                X_num = X.reindex(columns=numeric_cols)
                if self.numeric_means_ is not None:
                    X_num = X_num.fillna(self.numeric_means_)
                X[numeric_cols] = self.scaler.transform(X_num)

        X_enc = pd.get_dummies(X, drop_first=False)
        return X_enc.reindex(columns=self.feature_columns_, fill_value=0)

    def _prepare_inference_X(self, X):
        X = self._ensure_df(X)
        X = self._apply_inference_fairness(X)
        return self._encode_inference(X)

    def fit(self, X_train, y_train, sample_weight=None):
        X_train = self._ensure_df(X_train)
        y_train = self._ensure_series(y_train)

        X_train, y_train, sample_weight = self._apply_train_fairness(X_train, y_train, sample_weight)
        X_train = self._encode_train(X_train)

        if sample_weight is not None:
            self.base_model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.base_model.fit(X_train, y_train)

        self.classes_ = self.base_model.classes_

        return self

    def predict(self, X):
        X_prepared = self._prepare_inference_X(X)
        return self.base_model.predict(X_prepared)

    def predict_proba(self, X):
        X_prepared = self._prepare_inference_X(X)
        return self.base_model.predict_proba(X_prepared)


def print_config(config):
    for key, value in config.items():
        logger.info("  %s: %s", key, value if not isinstance(value, Enum) else value.value)


def training_pipeline(adult_df: pd.DataFrame, config: dict):
    logger.info("Starting cv training pipeline with config:")
    print_config(config)

    target_col = config.get("target_col", "income")
    sensitive_cols = config.get("sensitive_cols", ["sex", "race"])

    clean_df = prepare_raw_df(
        adult_df,
        missing_strategy=config.get("missing_strategy", MissingStrategy.UNKNOWN),
        education_mode=config.get("education_mode", EducationMode.NUM),
        target_col=target_col,
    )
    logger.info("Data prepared: %s rows, %s columns", clean_df.shape[0], clean_df.shape[1])

    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col].astype(int)
    sensitive_features = X[sensitive_cols].copy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_type = config.get("model_type", Model.GB)
    # These were found to be the best parameters for GradientBoostingClassifier by a RandomizedSearchCV run with the parameter distributions defined below
    model_params = {
        "subsample": 1.0, 
        "n_estimators": 500, 
        "min_samples_split": 5, 
        "min_samples_leaf": 2, 
        "max_depth": 5, 
        "learning_rate": 0.05
    }
    model_params = config.get("model_params", model_params)
    
    if model_type == Model.LogReg:
        base_model = LogisticRegression(random_state=42, **model_params)
    elif model_type == Model.RF:
        base_model = RandomForestClassifier(random_state=42, **model_params)
    elif model_type == Model.GB:
        base_model = GradientBoostingClassifier(random_state=42, **model_params)

    model = MetaModel(
        base_model=base_model,
        fairness_mode=config.get("fairness_mode", FairnessMode.NONE),
        sensitive_cols=sensitive_cols,
    )

    predictions = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    logger.info("Training pipeline completed")

    if config.get("train_final_model", False):
        logger.info("Fitting final model on entire dataset")
        model.fit(X, y)

    return predictions, y, sensitive_features, model if config.get("train_final_model", False) else None


def evaluation_pipeline(y_true, y_pred_proba, sensitive_features):
    y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba[:, 1])

    demographic_parity_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    demographic_parity_r = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
    equalized_odds_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    equalized_odds_r = equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_features)

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "demographic_parity_difference": round(demographic_parity_diff, 4),
        "demographic_parity_ratio": round(demographic_parity_r, 4),
        "equalized_odds_difference": round(equalized_odds_diff, 4),
        "equalized_odds_ratio": round(equalized_odds_r, 4),
    }

    logger.info("Evaluation results:")
    for metric, value in metrics.items():
        logger.info("  %s: %s", metric, value)

    return metrics

def sklearn_hpo(adult_df: pd.DataFrame, config: dict, param_distributions: dict):
    logger.info("Starting sklearn HPO pipeline with config:")
    print_config(config)

    target_col = config.get("target_col", "income")
    sensitive_cols = config.get("sensitive_cols", ["sex", "race"])

    clean_df = prepare_raw_df(
        adult_df,
        missing_strategy=config.get("missing_strategy", MissingStrategy.UNKNOWN),
        education_mode=config.get("education_mode", EducationMode.NUM),
        target_col=target_col,
    )
    logger.info("Data prepared: %s rows, %s columns", clean_df.shape[0], clean_df.shape[1])

    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col].astype(int)

    base_model = GradientBoostingClassifier(random_state=42)
    model = MetaModel(
        base_model=base_model,
        fairness_mode=config.get("fairness_mode", FairnessMode.NONE),
        sensitive_cols=sensitive_cols,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
    )

    logger.info("Running RandomSearchCV...")
    search.fit(X, y)

    logger.info("Best parameters found: %s", search.best_params_)
    logger.info("Best cross-validation AUC: %s", search.best_score_)

    return search


if __name__ == "__main__":
    adult_df = pd.read_csv("data/adult.csv")

    config = {
        "missing_strategy": MissingStrategy.UNKNOWN,
        "education_mode": EducationMode.NUM,
        "fairness_mode": FairnessMode.NONE,
        "target_col": "income",
        "sensitive_cols": ["sex", "race"],
        "model_type": Model.GB,
        "train_final_model": False,  # Whether to fit a final model on the entire dataset after CV (useful for MLflow logging)
        # These were found to be the best parameters for GradientBoostingClassifier by a RandomizedSearchCV run with the parameter distributions defined below
        "model_params": {
            "subsample": 1.0, 
            "n_estimators": 500, 
            "min_samples_split": 5, 
            "min_samples_leaf": 2, 
            "max_depth": 5, 
            "learning_rate": 0.05
        }
    }

    result, y_true, sensitive_features, model = training_pipeline(adult_df, config=config)
    metrics = evaluation_pipeline(y_true, result, sensitive_features)


    param_distributions = {
        "base_model__n_estimators": [100, 200, 300, 500],
        "base_model__learning_rate": [0.01, 0.05, 0.001, 0.005],
        "base_model__max_depth": [3, 5, 7, 9],
        "base_model__min_samples_split": [2, 5, 10],
        "base_model__min_samples_leaf": [1, 2, 3],
        "base_model__subsample": [0.8, 0.9, 1.0]
    }

    # hpo_result = sklearn_hpo(adult_df, config=config, param_distributions=param_distributions)
