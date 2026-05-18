import numpy as np
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)


# =========================
# FEATURES
# =========================
FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "native.country"
]


# =========================
# INDEX
# =========================
@app.route("/")
def index():
    return render_template("index.html")


# =========================
# DATA TRANSFORM
# =========================
def safe_float(value):
    try:
        return float(value)
    except:
        return None


def transform(data):
    X = []

    for row in data:

        X.append([
            safe_float(row.get("age")),
            row.get("workclass", ""),
            safe_float(row.get("fnlwgt")),
            row.get("education", ""),
            row.get("marital.status", ""),
            row.get("occupation", ""),
            row.get("relationship", ""),
            row.get("race", ""),
            row.get("sex", ""),
            safe_float(row.get("capital.gain")),
            safe_float(row.get("capital.loss")),
            safe_float(row.get("hours.per.week")),
            row.get("native.country", "")
        ])

    print(np.array(X))

    return np.array(X)


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

        model = body.get("model")

        if not model:
            return jsonify({"error": "No model provided"}), 400

        X = transform(data)
        preds = np.ones(len(X))

        return jsonify({
            "predictions": preds.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(debug=True)

