from pathlib import Path
import re
import joblib
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "spam_classifier.joblib"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024  # 20 KB request limit

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)

model = load_model()

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "")
    if not isinstance(message, str):
        return jsonify({"error": "Message must be text."}), 400

    message = message.strip()
    if not message:
        return jsonify({"error": "Please enter an email message."}), 400
    if len(message) > 10000:
        return jsonify({"error": "Message is too long (maximum 10,000 characters)."}), 400

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(prediction)]) * 100

    # Return probabilities for both classes so the UI can show the model's estimate.
    scores = {str(label): round(float(prob) * 100, 2)
              for label, prob in zip(classes, probabilities)}

    return jsonify({
        "label": str(prediction),
        "display_label": "Spam" if prediction == "spam" else "Not Spam",
        "confidence": round(confidence, 2),
        "probabilities": scores
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True})

if __name__ == "__main__":
    # Debug is disabled by default; enable only on your own development machine.
    app.run(host="127.0.0.1", port=5000, debug=False)
