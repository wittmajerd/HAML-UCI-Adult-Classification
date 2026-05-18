import numpy as np
from flask import Flask, request, jsonify, render_template
import mlflow
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

MLFLOW_TRACKING_URI = "file:///app/mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# =========================
# INDEX
# =========================
@app.route("/")
def index():
    return render_template("index.html")


# =========================
# PREDICT ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()

        data = body.get("data", [])

        if not data:
            return jsonify({"error": "No data provided"}), 400

        model_name = body.get("model")

        if not model_name:
            return jsonify({"error": "No model provided"}), 400

        model = load_model(model_name)
        if not model:
            return jsonify({"error": "Model not found"}), 404

        X = pd.DataFrame(X)
        X = X.replace("", "?")

        prob_preds = model.predict(X) #np.ones(len(X))
        threshold = 0.5
        preds = (prob_preds >= threshold).astype(int)
        app.logger.info(f"Making predictions with model '{model_name}' on {len(X)} samples")
        return jsonify({
            "predictions": preds.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def load_model(model_name: str):
    try:
        app.logger.info(f"Attempting to load model: {model_name}")
        model_uri = f"models:/{model_name}@final"
        app.logger.info(f"Loading model: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        app.logger.error(f"Failed to load model '{model_name}': {e}")
        return None
    

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=5555,  threaded=False)

