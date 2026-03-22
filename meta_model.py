from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


TARGET_COL = "income"
SENSITIVE_COLS = ["sex", "race"]


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

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.replace(".", "", regex=False)

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

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL] == ">50K"

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


# Base sklearn model inheritor - move preprocessing into fit
# sklearn pipeline can be used
# 1 stratified kfold
# 2 training but we do the prepocesseng steps in the fit method of the meta model, 
# so we can do the reweighing and onehot encoding there, 
# and then call the base model fit with the processed data and sample weights if needed
# this way the order of preprocessing steps is preserved and we can easily switch between different fairness modes without changing the fold preparation logic
# 3 evaluation - sklearn pipeline takes care of this
class MetaModel:
    def __init__(
        self,
        base_model,
        fairness_mode: FairnessMode = FairnessMode.NONE,
        sensitive_cols=None,
        label_encode=False,
        scale_numeric=False,
        one_hot_encode=False,
    ):
        if not isinstance(fairness_mode, FairnessMode):
            raise TypeError("fairness_mode must be FairnessMode enum")

        self.base_model = base_model
        self.fairness_mode = fairness_mode
        self.sensitive_cols = list(sensitive_cols) if sensitive_cols is not None else list(SENSITIVE_COLS)
        self.feature_columns_ = None
        # TODO implement these options
        self.label_encode = label_encode
        self.scale_numeric = scale_numeric
        self.one_hot_encode = one_hot_encode

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
        X_train_enc = pd.get_dummies(X_train, drop_first=False)
        self.feature_columns_ = X_train_enc.columns.tolist()
        return X_train_enc

    def _encode_inference(self, X):
        if self.feature_columns_ is None:
            raise RuntimeError("MetaModel must be fitted before inference")

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

        return self

    def predict(self, X):
        X_prepared = self._prepare_inference_X(X)
        return self.base_model.predict(X_prepared)

    def predict_proba(self, X):
        X_prepared = self._prepare_inference_X(X)
        return self.base_model.predict_proba(X_prepared)



if __name__ == "__main__":
    adult_df = pd.read_csv("data/adult.csv")

    clean_df = prepare_raw_df(
        adult_df,
        missing_strategy=MissingStrategy.UNKNOWN,
        education_mode=EducationMode.NUM,
    )

    X = clean_df.drop(columns=[TARGET_COL])
    y = clean_df[TARGET_COL].astype(int)

    # Ez az egész lehet sklearn pipeline is a MetaModel custom modellel
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        model = MetaModel(
            base_model=LogisticRegression(max_iter=1000),
            fairness_mode=FairnessMode.REWEIGH,
            sensitive_cols=SENSITIVE_COLS,
            label_encode=False,
            scale_numeric=False,
            one_hot_encode=True,
        )

        model.fit(X_train, y_train)
        score = model.base_model.score(model._prepare_inference_X(X_test), y_test)
        print(f"Fold {fold_idx}: accuracy={score:.4f}")
