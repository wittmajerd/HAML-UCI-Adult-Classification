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

        X = pd.DataFrame(data)
        X = X.replace("", "?")

        preds = model.predict(X)[:, 1] #np.ones(len(X))

        return jsonify({
            "predictions": preds.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=5555,  threaded=False)

