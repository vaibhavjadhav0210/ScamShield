"""Abstract base class for all ScamShield detectors.

OOP Concepts Demonstrated:
    - Abstraction: `Detector` inherits from `abc.ABC` and defines the
      abstract `detect()` contract.
    - Polymorphism: Every concrete detector (`RuleDetector`, `MLDetector`,
      `HybridDetector`) implements `detect(message) -> DetectionResult` so
      `ScamShieldService` can swap strategies seamlessly.
    - Design Pattern (Strategy): `Detector` serves as the Strategy interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from scamshield.models.message import DetectionResult, Message


class Detector(ABC):
    """Abstract Strategy interface for scam detection engines."""

    @property
    def name(self) -> str:
        """Return the detector's class name for logging and attribution."""
        return self.__class__.__name__

    @abstractmethod
    def detect(self, message: Message) -> DetectionResult:
        """Analyze a `Message` and return an immutable `DetectionResult`.

        Args:
            message: A validated `Message` instance (`SMS`, `WhatsApp`, or
                `CallTranscript`).

        Returns:
            An immutable `DetectionResult` containing verdict, scam type,
            risk score (0-100), confidence, and supporting evidence.
        """
