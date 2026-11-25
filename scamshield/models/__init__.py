"""Model exports for ScamShield."""

from scamshield.models.message import (
    SCAM_CATEGORIES,
    SMS,
    CallTranscript,
    DetectionResult,
    Message,
    MessageFactory,
    WhatsApp,
)

__all__ = [
    "SCAM_CATEGORIES",
    "DetectionResult",
    "Message",
    "SMS",
    "WhatsApp",
    "CallTranscript",
    "MessageFactory",
]
