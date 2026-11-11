from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "emails.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str).str.lower().str.strip()

    allowed = {"spam", "ham"}
    if not set(data["label"]).issubset(allowed):
        raise ValueError("Labels must be either 'spam' or 'ham'.")
    if data["label"].nunique() != 2:
        raise ValueError("Training data must contain both spam and ham examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.20,
        random_state=42, stratify=data["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english",
            ngram_range=(1, 2), max_features=10000,
            strip_accents="unicode"
        )),
        ("classifier", MultinomialNB(alpha=0.5))
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print(f"Rows: {len(data)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Hold-out accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels: ham, spam):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    output = MODEL_DIR / "spam_classifier.joblib"
    joblib.dump(pipeline, output)
    print(f"\nSaved model to: {output}")

if __name__ == "__main__":
    main()
