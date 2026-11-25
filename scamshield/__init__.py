"""ScamShield: Indian Scam Message & Call Transcript Detector."""

from scamshield.detectors import Detector, HybridDetector, MLDetector, RuleDetector
from scamshield.exceptions import EmptyMessageError, InvalidMessageTypeError, ModelNotTrainedError, ScamShieldError
from scamshield.explain import Explainer, LLMExplainer, TemplateExplainer
from scamshield.history import HistoryLog
from scamshield.models import (
    SCAM_CATEGORIES,
    SMS,
    CallTranscript,
    DetectionResult,
    Message,
    MessageFactory,
    WhatsApp,
)
from scamshield.preprocess import TextCleaner
from scamshield.service import ScamShieldService

__all__ = [
    "SCAM_CATEGORIES",
    "ScamShieldError",
    "EmptyMessageError",
    "ModelNotTrainedError",
    "InvalidMessageTypeError",
    "DetectionResult",
    "Message",
    "SMS",
    "WhatsApp",
    "CallTranscript",
    "MessageFactory",
    "TextCleaner",
    "Detector",
    "RuleDetector",
    "MLDetector",
    "HybridDetector",
    "Explainer",
    "TemplateExplainer",
    "LLMExplainer",
    "HistoryLog",
    "ScamShieldService",
]
