"""Application service coordinating detection, explanation, and history logging.

OOP Concepts Demonstrated:
    - Composition: `ScamShieldService` HAS a `Detector`, an `Explainer`, and a
      `HistoryLog`.
    - Design Pattern (Strategy): Detectors and Explainers can be swapped at
      runtime via `set_detector()` and `set_explainer()` without changing
      service logic.
    - Design Pattern (Factory): Delegates message instantiation to
      `MessageFactory.create()`.
"""

from __future__ import annotations

from typing import Any

from scamshield.detectors.base import Detector
from scamshield.explain.explainer import Explainer, LLMExplainer
from scamshield.history import HistoryLog
from scamshield.models.message import DetectionResult, Message, MessageFactory


class ScamShieldService:
    """High-level facade orchestrating message creation, detection, explanation, and logging."""

    def __init__(
        self,
        detector: Detector,
        explainer: Explainer | None = None,
        history: HistoryLog | None = None,
    ) -> None:
        self._detector: Detector = detector
        self._explainer: Explainer = explainer or LLMExplainer()
        self._history: HistoryLog = history or HistoryLog()

    @property
    def detector(self) -> Detector:
        """Return the active detection strategy."""
        return self._detector

    @property
    def explainer(self) -> Explainer:
        """Return the active explanation strategy."""
        return self._explainer

    @property
    def history(self) -> HistoryLog:
        """Return the HistoryLog instance."""
        return self._history

    def set_detector(self, detector: Detector) -> None:
        """Swap the active `Detector` strategy at runtime."""
        self._detector = detector

    def set_explainer(self, explainer: Explainer) -> None:
        """Swap the active `Explainer` strategy at runtime."""
        self._explainer = explainer

    def analyze(
        self,
        msg_type: str,
        text: str,
        save_history: bool = True,
        **kwargs: Any,
    ) -> tuple[DetectionResult, str]:
        """Run end-to-end scam analysis on raw user input.

        Args:
            msg_type: Channel type ('sms', 'whatsapp', 'call_transcript').
            text: Raw message or call transcript text.
            save_history: Whether to append this check to `HistoryLog`.
            **kwargs: Optional channel metadata forwarded to `MessageFactory`.

        Returns:
            Tuple of `(DetectionResult, explanation_text)`.
        """
        message: Message = MessageFactory.create(msg_type=msg_type, text=text, **kwargs)
        result: DetectionResult = self._detector.detect(message)
        explanation: str = self._explainer.explain(message, result)
        if save_history:
            self._history.record(message, result)
        return result, explanation
