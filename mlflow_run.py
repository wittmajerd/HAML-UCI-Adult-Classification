from enum import Enum
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from pipeline import (
    MissingStrategy,
    EducationMode,
    FairnessMode,
    Model,
    training_pipeline,
    evaluation_pipeline,
)

def ml_flow_train(adult_df: pd.DataFrame, config: dict):
    mlflow.set_tracking_uri("http://host.docker.internal:5000") # <-- This line is if we are using the dev container
    # mlflow.set_tracking_uri("http://localhost:5000") # <-- This line is if we are running the script locally
    mlflow.set_experiment("Default")
    # Enable autologging for scikit-learn
    # mlflow.sklearn.autolog()

    with mlflow.start_run():
        y_score, y_true, sensitive_features, model = training_pipeline(adult_df, config=config)
        metrics = evaluation_pipeline(y_true, y_score, sensitive_features)

        if config.get("train_final_model", False):
            mlflow.sklearn.log_model(model, pyfunc_predict_fn="predict_proba", name="model")

        for key, value in config.items():
            mlflow.log_param(key, value if not isinstance(value, Enum) else value.value)

        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        _log_curves_and_curve_metrics(y_true, y_score[:, 1])


def _log_curves_and_curve_metrics(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    mlflow.log_metric("average_precision", round(ap, 4))

    roc_fig, roc_ax = plt.subplots(figsize=(6, 5))
    roc_ax.plot(fpr, tpr, label="ROC")
    roc_ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("ROC Curve")
    roc_ax.legend(loc="lower right")
    mlflow.log_figure(roc_fig, "roc_curve.png")
    plt.close(roc_fig)

    pr_fig, pr_ax = plt.subplots(figsize=(6, 5))
    pr_ax.plot(recall_vals, precision_vals, label="PR")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision-Recall Curve")
    pr_ax.legend(loc="lower left")
    mlflow.log_figure(pr_fig, "pr_curve.png")
    plt.close(pr_fig)

    hist_fig, hist_ax = plt.subplots(figsize=(6, 5))
    hist_ax.hist(y_score[y_true == 0], bins=20, alpha=0.5, label="Negative Class")
    hist_ax.hist(y_score[y_true == 1], bins=20, alpha=0.5, label="Positive Class")
    hist_ax.set_xlabel("Predicted Probability")
    hist_ax.set_ylabel("Frequency")
    hist_ax.set_title("Predicted Probability Distribution")
    hist_ax.legend(loc="upper center")
    mlflow.log_figure(hist_fig, "predicted_probability_histogram.png")
    plt.close(hist_fig)

    cm = confusion_matrix(y_true, y_score >= 0.5)
    cm_fig, cm_ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=cm_ax, values_format="d")
    cm_ax.set_title("Confusion Matrix")
    mlflow.log_figure(cm_fig, "confusion_matrix.png")
    plt.close(cm_fig)


def ml_flow_inference(run_id: str, input_data: pd.DataFrame):
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    predictions = model.predict(input_data)
    return predictions


def grid_search_ml_flow(adult_df: pd.DataFrame, param_grid: dict):
    for missing_strategy in param_grid["missing_strategy"]:
        for education_mode in param_grid["education_mode"]:
            for fairness_mode in param_grid["fairness_mode"]:
                for model_type in param_grid["model_type"]:
                    config = {
                        "missing_strategy": missing_strategy,
                        "education_mode": education_mode,
                        "fairness_mode": fairness_mode,
                        "model_type": model_type,
                        "target_col": "income",
                        "sensitive_cols": ["sex", "race"],
                        "train_final_model": False,  # Don't train final model for grid search runs
                    }
                    ml_flow_train(adult_df, config=config)


if __name__ == "__main__":
    adult_df = pd.read_csv("data/adult.csv")
    # config = {
    #     "missing_strategy": MissingStrategy.UNKNOWN,
    #     "education_mode": EducationMode.NUM,
    #     "fairness_mode": FairnessMode.NONE,
    #     "model_type": Model.GB,
    #     "target_col": "income",
    #     "sensitive_cols": ["sex", "race"],
    #     "train_final_model": True,  # Whether to fit a final model on the entire dataset after CV (useful for MLflow logging)
    # }

    # ml_flow_train(adult_df, config=config)

    # Full param grid
    param_grid = {
        "missing_strategy": [MissingStrategy.UNKNOWN, MissingStrategy.DROP],
        "education_mode": [EducationMode.NUM, EducationMode.CAT, EducationMode.BOTH],
        "fairness_mode": [FairnessMode.NONE, FairnessMode.REWEIGH, FairnessMode.DROP, FairnessMode.MASK],
        "model_type": [Model.LogReg, Model.RF, Model.GB],
    }

    # param_grid = {
    #     "missing_strategy": [MissingStrategy.UNKNOWN, MissingStrategy.DROP],
    #     "education_mode": [EducationMode.NUM, EducationMode.CAT, EducationMode.BOTH],
    #     "fairness_mode": [FairnessMode.NONE, FairnessMode.REWEIGH],
    #     "model_type": [ Model.GB],
    # }
    grid_search_ml_flow(adult_df, param_grid=param_grid)