"""Machine-learning scam detector using TF-IDF (word + char n-grams) and LogisticRegression.

OOP Concepts Demonstrated:
    - Inheritance: `MLDetector` inherits from `Detector`.
    - Encapsulation: Model pipeline, cleaner, and trained state (`_pipeline`,
      `_is_trained`) are private attributes accessed via controlled methods.
    - Custom Exceptions: Raises `ModelNotTrainedError` when `detect()` is
      invoked before training or loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from scamshield.detectors.base import Detector
from scamshield.exceptions import ModelNotTrainedError
from scamshield.models.message import DetectionResult, Message
from scamshield.preprocess import TextCleaner

DEFAULT_MODEL_PATH = Path("models_artifacts") / "ml_detector.joblib"


class MLDetector(Detector):
    """Multi-class TF-IDF + LogisticRegression detector for 6 scam categories."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._cleaner: TextCleaner = TextCleaner(preserve_placeholders=True)
        self._pipeline: Pipeline | None = None
        self._is_trained: bool = False
        self._model_path: Path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if model_path is not None and self._model_path.exists():
            self.load(self._model_path)

    @property
    def is_trained(self) -> bool:
        """Return True if the underlying scikit-learn pipeline is fitted."""
        return self._is_trained and self._pipeline is not None

    @property
    def classes(self) -> tuple[str, ...]:
        """Return the ordered class labels learned by the classifier."""
        if not self.is_trained or self._pipeline is None:
            raise ModelNotTrainedError()
        clf: LogisticRegression = self._pipeline.named_steps["clf"]
        return tuple(str(c) for c in clf.classes_)

    @staticmethod
    def build_pipeline() -> Pipeline:
        """Construct the word + char_wb TF-IDF + LogisticRegression pipeline."""
        features = FeatureUnion(
            transformer_list=[
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=8000,
                        sublinear_tf=True,
                        min_df=2,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=8000,
                        sublinear_tf=True,
                        min_df=3,
                    ),
                ),
            ]
        )
        clf = LogisticRegression(
            max_iter=1000,
            C=2.0,
            class_weight="balanced",
            random_state=42,
        )
        return Pipeline([("features", features), ("clf", clf)])

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> MLDetector:
        """Clean texts and fit the TF-IDF + LogisticRegression pipeline."""
        cleaned = [self._cleaner.clean(t) for t in texts]
        y = list(labels)
        self._pipeline = self.build_pipeline()
        self._pipeline.fit(cleaned, y)
        self._is_trained = True
        return self

    def predict_proba_dict(self, message: Message) -> dict[str, float]:
        """Return class -> probability mapping for `message`."""
        if not self.is_trained or self._pipeline is None:
            raise ModelNotTrainedError()
        cleaned = self._cleaner.clean(message.text)
        probs = self._pipeline.predict_proba([cleaned])[0]
        clf: LogisticRegression = self._pipeline.named_steps["clf"]
        return {str(cls): float(prob) for cls, prob in zip(clf.classes_, probs)}

    def _extract_top_features(self, cleaned_text: str, class_idx: int, top_k: int = 5) -> tuple[str, ...]:
        """Extract the highest-contributing word n-grams for the predicted class."""
        if self._pipeline is None:
            return ()
        union: FeatureUnion = self._pipeline.named_steps["features"]
        clf: LogisticRegression = self._pipeline.named_steps["clf"]

        word_vec: TfidfVectorizer = union.named_transformers["word_tfidf"]
        word_matrix = word_vec.transform([cleaned_text])
        word_names = word_vec.get_feature_names_out()
        n_word = len(word_names)

        # LogisticRegression coef_ shape: (n_classes, n_features)
        coef_word = clf.coef_[class_idx][:n_word]
        contributions = word_matrix.toarray()[0] * coef_word
        positive_indices = np.where(contributions > 0)[0]
        if len(positive_indices) == 0:
            return ()
        sorted_idx = positive_indices[np.argsort(contributions[positive_indices])[::-1]]
        return tuple(str(word_names[i]) for i in sorted_idx[:top_k])

    def detect(self, message: Message) -> DetectionResult:
        """Predict scam category and risk score using the trained ML pipeline."""
        if not self.is_trained or self._pipeline is None:
            raise ModelNotTrainedError()

        cleaned = self._cleaner.clean(message.text)
        probs = self._pipeline.predict_proba([cleaned])[0]
        clf: LogisticRegression = self._pipeline.named_steps["clf"]
        classes = [str(c) for c in clf.classes_]

        pred_idx = int(np.argmax(probs))
        pred_label = classes[pred_idx]
        pred_prob = float(probs[pred_idx])

        # Scam probability is 1 - P(Safe)
        if "Safe" in classes:
            safe_idx = classes.index("Safe")
            scam_prob = 1.0 - float(probs[safe_idx])
        else:
            scam_prob = pred_prob if pred_label != "Safe" else 1.0 - pred_prob

        risk_score = int(round(max(0.0, min(100.0, scam_prob * 100.0))))
        verdict = "SAFE" if pred_label == "Safe" else "SCAM"
        top_features = self._extract_top_features(cleaned, pred_idx, top_k=5)

        return DetectionResult(
            verdict=verdict,
            scam_type=pred_label,
            risk_score=risk_score,
            confidence=round(pred_prob, 4),
            detector_name=self.name,
            triggered_rules=(),
            rule_reasons=(),
            top_features=top_features,
        )

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the trained pipeline to disk using joblib."""
        if not self.is_trained or self._pipeline is None:
            raise ModelNotTrainedError("Cannot save an untrained MLDetector.")
        target = Path(path) if path else self._model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "cleaner": self._cleaner}, target)
        return target

    def load(self, path: str | Path | None = None) -> MLDetector:
        """Load a previously trained pipeline from disk."""
        source = Path(path) if path else self._model_path
        if not source.exists():
            raise FileNotFoundError(f"Model file not found: {source}")
        payload = joblib.load(source)
        self._pipeline = payload["pipeline"]
        self._cleaner = payload.get("cleaner", TextCleaner(preserve_placeholders=True))
        self._is_trained = True
        return self
