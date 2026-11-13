from functools import lru_cache
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .data import load


def fit_baseline(rows):
    training = [r for r in rows if r["split"] == "train"]
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(C=3, max_iter=1000, random_state=42),
    )
    model.fit([r["text"] for r in training], [r["label"] for r in training])
    return model


@lru_cache
def baseline():
    return fit_baseline(load())


def classify(text):
    model = baseline()
    probabilities = model.predict_proba([text])[0]
    index = probabilities.argmax()
    return {
        "label": str(model.classes_[index]),
        "confidence": float(probabilities[index]),
        "scores": dict(zip(model.classes_, map(float, probabilities))),
        "review_required": bool(probabilities[index] < 0.6),
        "model": "tfidf-logistic-regression",
    }
