import math
import pickle
from pathlib import Path

import numpy as np

from preprocess import preprocess


class ModelUnavailableError(RuntimeError):
    pass


class MLService:
    """Adapter around the team's existing TF-IDF + Linear SVM artifacts."""

    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parents[2]
        self._vectorizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None and self._vectorizer is not None:
            return
        models_dir = self.root / "Models"
        vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
        model_path = models_dir / "linear_svm.pkl"
        if not vectorizer_path.exists() or not model_path.exists():
            raise ModelUnavailableError("Model artifacts are missing from the Models directory.")
        with vectorizer_path.open("rb") as vectorizer_file:
            self._vectorizer = pickle.load(vectorizer_file)
        with model_path.open("rb") as model_file:
            self._model = pickle.load(model_file)

    def predict(self, title: str | None, body: str) -> tuple[str, float]:
        self._load()
        cleaned = preprocess(title or "", body)
        if not cleaned.strip():
            raise ValueError("No usable text remains after preprocessing.")
        vector = self._vectorizer.transform([cleaned])
        prediction = int(self._model.predict(vector)[0])
        decision = float(self._model.decision_function(vector)[0])
        # The original Streamlit app uses a sigmoid of the SVM margin as a
        # relative confidence signal. Retain that behaviour, without calling
        # it a calibrated probability.
        clipped = max(min(decision, 500), -500)
        probability_fake = 1.0 / (1.0 + math.exp(-clipped))
        confidence = probability_fake if prediction == 1 else 1.0 - probability_fake
        return ("FAKE" if prediction == 1 else "REAL"), float(np.clip(confidence, 0, 1))
