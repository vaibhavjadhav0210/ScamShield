"""Hybrid scam detector combining MLDetector and RuleDetector.

OOP Concepts Demonstrated:
    - Inheritance: `HybridDetector` inherits from `Detector`.
    - Composition: `HybridDetector` HAS a `RuleDetector` and an `MLDetector`
      (delegating to both and fusing their scores).
    - Encapsulation: Weights (`_ml_weight`, `_rule_weight`) are validated and
      protected as private attributes.
"""

from __future__ import annotations

from scamshield.detectors.base import Detector
from scamshield.detectors.ml import MLDetector
from scamshield.detectors.rule import RuleDetector
from scamshield.models.message import DetectionResult, Message


class HybridDetector(Detector):
    """Ensemble detector combining ML probability scores and heuristic rules.

    Default weighting: 0.6 * ML score + 0.4 * Rule score.
    """

    def __init__(
        self,
        ml_detector: MLDetector,
        rule_detector: RuleDetector | None = None,
        ml_weight: float = 0.6,
        rule_weight: float = 0.4,
        scam_threshold: int = 45,
    ) -> None:
        if not abs((ml_weight + rule_weight) - 1.0) < 1e-6:
            raise ValueError("ml_weight and rule_weight must sum to 1.0.")
        # Composition: HybridDetector owns/composes an MLDetector and a RuleDetector
        self._ml_detector: MLDetector = ml_detector
        self._rule_detector: RuleDetector = rule_detector or RuleDetector()
        self._ml_weight: float = ml_weight
        self._rule_weight: float = rule_weight
        self._scam_threshold: int = scam_threshold

    @property
    def ml_detector(self) -> MLDetector:
        """Return the composed MLDetector."""
        return self._ml_detector

    @property
    def rule_detector(self) -> RuleDetector:
        """Return the composed RuleDetector."""
        return self._rule_detector

    @property
    def ml_weight(self) -> float:
        """Return the weight assigned to MLDetector."""
        return self._ml_weight

    @property
    def rule_weight(self) -> float:
        """Return the weight assigned to RuleDetector."""
        return self._rule_weight

    def detect(self, message: Message) -> DetectionResult:
        """Fuse ML and rule-based detections into a single `DetectionResult`."""
        ml_res = self._ml_detector.detect(message)
        rule_res = self._rule_detector.detect(message)

        combined_score = int(
            round(self._ml_weight * ml_res.risk_score + self._rule_weight * rule_res.risk_score)
        )
        combined_score = max(0, min(100, combined_score))

        # Determine final verdict and scam_type
        if combined_score >= self._scam_threshold or (
            ml_res.is_scam and ml_res.confidence >= 0.70
        ):
            verdict = "SCAM"
            if ml_res.scam_type != "Safe":
                scam_type = ml_res.scam_type
            elif rule_res.scam_type != "Safe":
                scam_type = rule_res.scam_type
            else:
                scam_type = "Fake KYC/Bank"
        else:
            verdict = "SAFE"
            scam_type = "Safe"

        confidence = round(
            max(
                0.50,
                min(
                    0.99,
                    self._ml_weight * ml_res.confidence + self._rule_weight * rule_res.confidence,
                ),
            ),
            4,
        )

        return DetectionResult(
            verdict=verdict,
            scam_type=scam_type,
            risk_score=combined_score,
            confidence=confidence,
            detector_name=self.name,
            triggered_rules=rule_res.triggered_rules,
            rule_reasons=rule_res.rule_reasons,
            top_features=ml_res.top_features,
        )
