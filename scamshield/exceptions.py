"""Custom exceptions for the ScamShield application.

OOP Concept Demonstrated:
    - Inheritance & Custom Exception Hierarchy: All domain-specific exceptions
      inherit from a common `ScamShieldError` base class so callers can catch
      either specific errors or any ScamShield-level failure.
"""

from __future__ import annotations


class ScamShieldError(Exception):
    """Base exception class for all ScamShield domain errors."""


class EmptyMessageError(ScamShieldError, ValueError):
    """Raised when an empty or whitespace-only message is provided.

    Inherits from both `ScamShieldError` and `ValueError` for standard
    Pythonic validation compatibility.
    """

    def __init__(self, message: str = "Message text cannot be empty or whitespace.") -> None:
        super().__init__(message)


class ModelNotTrainedError(ScamShieldError, RuntimeError):
    """Raised when MLDetector.detect() is called before training or loading a model."""

    def __init__(
        self,
        message: str = "ML model is not trained or loaded. Run train.py or call load() first.",
    ) -> None:
        super().__init__(message)


class InvalidMessageTypeError(ScamShieldError, ValueError):
    """Raised when MessageFactory receives an unsupported message type."""

    def __init__(self, msg_type: str) -> None:
        super().__init__(
            f"Unsupported message type '{msg_type}'. "
            "Expected one of: 'sms', 'whatsapp', 'call_transcript'."
        )
