
from pathlib import Path
import joblib
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "spam_classifier.joblib"

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024

model = None

def load_model():
    global model

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )
        model = joblib.load(MODEL_PATH)

    return model


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        message = payload.get("message", "")

        if not isinstance(message, str):
            return jsonify({"error": "Message must be text."}), 400

        message = message.strip()

        if not message:
            return jsonify({"error": "Please enter an email message."}), 400

        if len(message) > 10000:
            return jsonify({"error": "Message is too long."}), 400

        clf = load_model()

        prediction = clf.predict([message])[0]
        probabilities = clf.predict_proba([message])[0]
        classes = list(clf.classes_)

        confidence = float(
            probabilities[classes.index(prediction)]
        ) * 100

        scores = {
            str(label): round(float(prob) * 100, 2)
            for label, prob in zip(classes, probabilities)
        }

        return jsonify({
            "label": str(prediction),
            "display_label": "Spam" if prediction == "spam" else "Not Spam",
            "confidence": round(confidence, 2),
            "probabilities": scores
        })

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check deployment logs."
        }), 500


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_exists": MODEL_PATH.exists()
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
