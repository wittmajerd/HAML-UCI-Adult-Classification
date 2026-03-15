import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from process import (
    prepare_raw_df, 
    build_stratified_folds, 
    apply_fairness_per_fold, 
    onehot_encode_prepared_folds, 
    MissingStrategy, 
    EducationMode, 
    FairnessMode, 
    TARGET_COL,
    SENSITIVE_COLS,
)

def ml_flow_train(X_train, y_train):
    mlflow.set_experiment("MLflow Quickstart")
    # Enable autologging for scikit-learn
    mlflow.sklearn.autolog()

    gbr = HistGradientBoostingClassifier(random_state=42)
    gbr.fit(X_train, y_train)

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


    ml_flow_train(ready_folds[0]['X_train'], ready_folds[0]['y_train'])