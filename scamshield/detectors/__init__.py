"""Detector exports for ScamShield."""

from scamshield.detectors.base import Detector
from scamshield.detectors.hybrid import HybridDetector
from scamshield.detectors.ml import MLDetector
from scamshield.detectors.rule import Rule, RuleDetector

__all__ = [
    "Detector",
    "Rule",
    "RuleDetector",
    "MLDetector",
    "HybridDetector",
]
