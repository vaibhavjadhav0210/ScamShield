"""Training script for ScamShield MLDetector.

Performs:
    1. Stratified 80/20 train/test split (random_state=42).
    2. 5-fold stratified cross-validation on the training split.
    3. Final fit on the 80% training split (leaving the 20% test split
       strictly untouched for `evaluate.py`).
    4. Persists the trained `MLDetector` to `models_artifacts/ml_detector.joblib`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from scamshield.detectors.ml import DEFAULT_MODEL_PATH, MLDetector
from scamshield.preprocess import TextCleaner

DATA_PATH = Path("data") / "messages.csv"


def train_model(
    data_path: Path = DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, float]:
    """Train MLDetector with 5-fold CV and save the fitted artifact."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run `python data/generate_dataset.py` first."
        )

    df = pd.read_csv(data_path)
    X_train, _X_test, y_train, _y_test = train_test_split(
        df["text"].astype(str),
        df["label"].astype(str),
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    cleaner = TextCleaner(preserve_placeholders=True)
    X_train_clean = [cleaner.clean(t) for t in X_train]

    cv_pipeline = MLDetector.build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        cv_pipeline,
        X_train_clean,
        y_train.tolist(),
        cv=cv,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        n_jobs=1,
    )

    cv_acc = float(np.mean(cv_scores["test_accuracy"]))
    cv_f1 = float(np.mean(cv_scores["test_f1_macro"]))

    detector = MLDetector(model_path=model_path)
    detector.fit(X_train.tolist(), y_train.tolist())
    saved_path = detector.save(model_path)

    print(f"Training samples: {len(X_train)} | Held-out test samples: {len(_X_test)}")
    print(f"5-Fold CV Accuracy : {cv_acc:.4f}")
    print(f"5-Fold CV Macro-F1 : {cv_f1:.4f}")
    print(f"Saved trained MLDetector to {saved_path}")

    return {"cv_accuracy": cv_acc, "cv_f1_macro": cv_f1}


if __name__ == "__main__":
    train_model()
