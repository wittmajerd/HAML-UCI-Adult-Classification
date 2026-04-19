from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate


class MissingStrategy(str, Enum):
    DROP = "drop"
    UNKNOWN = "unknown"


class EducationMode(str, Enum):
    NUM = "num"
    CAT = "cat"
    BOTH = "both"


class FairnessMode(str, Enum):
    NONE = "none"
    DROP = "drop"
    REWEIGH = "reweigh"
    MASK = "mask"


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

    if missing_strategy == MissingStrategy.DROP:
        df = df.dropna().reset_index(drop=True)
    else:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [col for col in df.columns if col not in num_cols]
        for col in cat_cols:
            df[col] = df[col].fillna("Unknown")
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())

    edu_num_col = "education_num" if "education_num" in df.columns else "education.num"
    edu_cat_col = "education"

    if education_mode == EducationMode.NUM and edu_cat_col in df.columns:
        df = df.drop(columns=[edu_cat_col])
    if education_mode == EducationMode.CAT and edu_num_col in df.columns:
        df = df.drop(columns=[edu_num_col])

    if target_col in df.columns:
        df[target_col] = df[target_col] == ">50K"

    return df


# ez most csake egy ilyen random implementáció, de kicserélhető akár aif360 reweightre
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


def training_pipeline(adult_df: pd.DataFrame, config: dict):
    clean_df = prepare_raw_df(
        adult_df,
        missing_strategy=config["missing_strategy"],
        education_mode=config["education_mode"],
        target_col=config["target_col"],
    )

    X = clean_df.drop(columns=[config["target_col"]])
    y = clean_df[config["target_col"]].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = MetaModel(
        # base_model=LogisticRegression(max_iter=1000, random_state=42),
        # base_model=RandomForestClassifier(n_estimators=100, random_state=42),
        base_model=config["model_type"](n_estimators=100, random_state=42),
        fairness_mode=config["fairness_mode"],
        sensitive_cols=config["sensitive_cols"],
    )

    result = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)

    return result, y

# TODO more metrics - for example, group-wise metrics for fairness evaluation
def evaluation_pipeline(y_true, y_pred_proba):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba[:, 1])

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")


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

    result, y_true = training_pipeline(adult_df, config=config)

    evaluation_pipeline(y_true, result)
