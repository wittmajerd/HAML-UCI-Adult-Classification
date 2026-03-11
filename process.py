from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd
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


def build_stratified_folds(df, target_col=TARGET_COL, n_splits=5, random_state=42):
    """Build StratifiedKFold splits on the full dataframe."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    y = df[target_col].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for train_idx, test_idx in skf.split(df, y):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        folds.append((train_df, test_df))

    return folds


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


def apply_fairness_per_fold(
    folds,
    target_col=TARGET_COL,
    sensitive_cols=SENSITIVE_COLS,
    fairness_mode: FairnessMode = FairnessMode.NONE,
):
    """
    For each fold, prepare train/test with optional fairness handling.

    fairness_mode:
    - 'none': no fairness preprocessing
    - 'drop': drop sensitive columns
    - 'reweigh': compute train-only sample weights
    - 'mask': no train-time action (reserved for inference-time masking workflows)
    """
    if not isinstance(fairness_mode, FairnessMode):
        raise TypeError("fairness_mode must be FairnessMode enum")

    prepared_folds = []

    for fold_idx, (train_df, test_df) in enumerate(folds, start=1):
        y_train = train_df[target_col].astype(int).reset_index(drop=True)
        y_test = test_df[target_col].astype(int).reset_index(drop=True)
        X_train = train_df.drop(columns=[target_col]).copy()
        X_test = test_df.drop(columns=[target_col]).copy()

        available_sensitive = [c for c in sensitive_cols if c in X_train.columns]
        sample_weight = None

        if fairness_mode == FairnessMode.REWEIGH:
            if available_sensitive:
                sensitive_train = X_train[available_sensitive].copy()
                sample_weight = _manual_reweighing(y_train, sensitive_train)
            else:
                sample_weight = np.ones(len(y_train), dtype=float)
        elif fairness_mode == FairnessMode.DROP and available_sensitive:
            X_train = X_train.drop(columns=available_sensitive)
            X_test = X_test.drop(columns=available_sensitive)

        prepared_folds.append(
            {
                "fold": fold_idx,
                "X_train": X_train.reset_index(drop=True),
                "y_train": y_train,
                "X_test": X_test.reset_index(drop=True),
                "y_test": y_test,
                "sample_weight": sample_weight,
            }
        )

    return prepared_folds


def onehot_encode_prepared_folds(prepared_folds):
    """One-hot encode X_train/X_test per fold and align columns using train as reference."""
    encoded_folds = []

    for fold_data in prepared_folds:
        X_train_enc = pd.get_dummies(fold_data["X_train"], drop_first=False)
        X_test_enc = pd.get_dummies(fold_data["X_test"], drop_first=False)
        X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

        updated = dict(fold_data)
        updated["X_train"] = X_train_enc.reset_index(drop=True)
        updated["X_test"] = X_test_enc.reset_index(drop=True)
        encoded_folds.append(updated)

    return encoded_folds


if __name__ == "__main__":
    adult_df = pd.read_csv("data/adult.csv")

    clean_df = prepare_raw_df(
        adult_df,
        missing_strategy=MissingStrategy.UNKNOWN,
        education_mode=EducationMode.NUM,
    )

    folds = build_stratified_folds(clean_df, target_col=TARGET_COL, n_splits=5, random_state=42)
    ready_folds = apply_fairness_per_fold(
        folds,
        target_col=TARGET_COL,
        sensitive_cols=SENSITIVE_COLS,
        fairness_mode=FairnessMode.NONE,
    )
    ready_folds = onehot_encode_prepared_folds(ready_folds)

    print("Num folds:", len(ready_folds))
    for fold_data in ready_folds:
        print(
            f"Fold {fold_data['fold']} - X_train shape: {fold_data['X_train'].shape}, "
            f"X_test shape: {fold_data['X_test'].shape}, "
            f"Train positive ratio: {fold_data['y_train'].mean().round(4)}"
        )
    # print("Sample fold columns:", ready_folds[0]["X_train"].columns.tolist())
