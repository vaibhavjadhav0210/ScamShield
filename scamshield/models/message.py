"""Domain models for ScamShield messages, factory, and immutable detection results.

OOP Concepts Demonstrated:
    - Abstraction: `Message` is an abstract base class (`abc.ABC`) defining the
      interface (`get_context_features`) that all message channels must implement.
    - Inheritance: `SMS`, `WhatsApp`, and `CallTranscript` inherit from `Message`.
    - Polymorphism: Each subclass overrides `get_context_features()` to extract
      channel-specific metadata while sharing a uniform interface.
    - Encapsulation: Private attributes (`_text`, `_sender`, etc.) with read-only
      `@property` accessors and constructor input validation; `DetectionResult` is
      an immutable (`frozen=True`) dataclass.
    - Design Pattern (Factory): `MessageFactory.create()` encapsulates object
      creation logic for different message channels.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from scamshield.exceptions import EmptyMessageError, InvalidMessageTypeError

SCAM_CATEGORIES: tuple[str, ...] = (
    "Digital Arrest",
    "Fake Courier",
    "Fake KYC/Bank",
    "Lottery/Prize",
    "Job/Investment",
    "Safe",
)

_URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(?:xyz|top|tk|ml|ga|cf|info|club|in|com)/\S*)", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?:\+?91[\s-]?)?[6-9]\d{9}\b")


@dataclass(frozen=True)
class DetectionResult:
    """Immutable value object representing the outcome of a scam detection check.

    OOP Concept: Encapsulation & Immutability (`frozen=True` prevents mutation
    after creation, ensuring auditability across detectors, explainers, and logs).
    """

    verdict: str  # "SCAM" or "SAFE"
    scam_type: str  # One of SCAM_CATEGORIES
    risk_score: int  # 0 to 100
    confidence: float  # 0.0 to 1.0
    detector_name: str
    triggered_rules: tuple[str, ...] = ()
    rule_reasons: tuple[str, ...] = ()
    top_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.verdict not in ("SCAM", "SAFE"):
            raise ValueError(f"verdict must be 'SCAM' or 'SAFE', got {self.verdict!r}")
        if not 0 <= self.risk_score <= 100:
            raise ValueError(f"risk_score must be in [0, 100], got {self.risk_score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

    @property
    def is_scam(self) -> bool:
        """Return True when the verdict is SCAM."""
        return self.verdict == "SCAM"


class Message(ABC):
    """Abstract base class representing a user-submitted communication.

    OOP Concept: Abstraction & Encapsulation. Stores validated text in a private
    attribute `_text` exposed via a read-only property.
    """

    def __init__(self, text: str, sender: str = "Unknown") -> None:
        if not isinstance(text, str) or not text.strip():
            raise EmptyMessageError()
        # Encapsulation: private attributes protected from direct external mutation.
        self._text: str = text.strip()
        self._sender: str = sender.strip() if sender and sender.strip() else "Unknown"

    @property
    def text(self) -> str:
        """Return the validated message text."""
        return self._text

    @property
    def sender(self) -> str:
        """Return the sender identifier (phone, header, or label)."""
        return self._sender

    @property
    def channel(self) -> str:
        """Return the human-readable channel name."""
        return self.__class__.__name__

    @abstractmethod
    def get_context_features(self) -> dict[str, Any]:
        """Extract channel-specific contextual signals for detection & explanation.

        OOP Concept: Polymorphism — subclasses override this method to provide
        channel-tailored context features.
        """

    def _base_signals(self) -> dict[str, Any]:
        """Compute common structural signals shared across all message types."""
        urls = _URL_PATTERN.findall(self._text)
        phones = _PHONE_PATTERN.findall(self._text)
        return {
            "channel": self.channel,
            "char_length": len(self._text),
            "word_count": len(self._text.split()),
            "has_url": len(urls) > 0,
            "url_count": len(urls),
            "has_phone": len(phones) > 0,
        }

    def __repr__(self) -> str:
        preview = self._text[:40] + ("..." if len(self._text) > 40 else "")
        return f"{self.channel}(sender={self._sender!r}, text={preview!r})"


class SMS(Message):
    """Represents a standard SMS text message (often with DLT/sender headers)."""

    def __init__(self, text: str, sender: str = "Unknown", has_short_code: bool = False) -> None:
        super().__init__(text=text, sender=sender)
        self._has_short_code: bool = has_short_code

    @property
    def has_short_code(self) -> bool:
        """Return whether the SMS originated from a short code or alpha header."""
        return self._has_short_code

    def get_context_features(self) -> dict[str, Any]:
        """Extract SMS-specific context such as header/short-code and link density."""
        features = self._base_signals()
        has_dlt_header = bool(re.search(r"\b[A-Z]{2}-[A-Z0-9]{4,8}\b", self._text))
        features.update(
            {
                "has_short_code": self._has_short_code,
                "has_dlt_header": has_dlt_header,
                "mentions_sms_reply": bool(re.search(r"\b(reply|stop|opt out)\b", self._text, re.I)),
            }
        )
        return features


class WhatsApp(Message):
    """Represents a WhatsApp message (often forwarded, with links or group invites)."""

    def __init__(
        self,
        text: str,
        sender: str = "Unknown",
        is_forwarded: bool = False,
    ) -> None:
        super().__init__(text=text, sender=sender)
        self._is_forwarded: bool = is_forwarded

    @property
    def is_forwarded(self) -> bool:
        """Return whether the message was marked as forwarded."""
        return self._is_forwarded

    def get_context_features(self) -> dict[str, Any]:
        """Extract WhatsApp-specific signals like group invites, Telegram redirects, and emojis."""
        features = self._base_signals()
        has_group_or_tg = bool(
            re.search(r"(chat\.whatsapp\.com|t\.me/|wa\.me/|telegram)", self._text, re.I)
        )
        has_emoji = bool(re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", self._text))
        features.update(
            {
                "is_forwarded": self._is_forwarded,
                "has_group_or_telegram_link": has_group_or_tg,
                "has_emoji": has_emoji,
            }
        )
        return features


class CallTranscript(Message):
    """Represents a transcribed phone or video call (common in Digital Arrest & KYC scams)."""

    def __init__(
        self,
        text: str,
        sender: str = "Unknown",
        duration_seconds: int = 0,
    ) -> None:
        super().__init__(text=text, sender=sender)
        if duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative.")
        self._duration_seconds: int = duration_seconds

    @property
    def duration_seconds(self) -> int:
        """Return the call duration in seconds if known."""
        return self._duration_seconds

    def get_context_features(self) -> dict[str, Any]:
        """Extract call-transcript cues such as speaker turns, IVR prompts, and video-call pressure."""
        features = self._base_signals()
        speaker_turns = len(re.findall(r"(?m)^(?:caller|agent|officer|victim|user|me)\s*:", self._text, re.I))
        mentions_video_or_stay_on_line = bool(
            re.search(
                r"(video\s*call|skype|whatsapp\s*video|don'?t\s*disconnect|stay\s*on\s*(?:the\s*)?line|press\s*\d)",
                self._text,
                re.I,
            )
        )
        features.update(
            {
                "duration_seconds": self._duration_seconds,
                "speaker_turns": speaker_turns,
                "mentions_video_or_ivr": mentions_video_or_stay_on_line,
            }
        )
        return features


class MessageFactory:
    """Factory class to instantiate the appropriate Message subclass.

    OOP Concept: Factory Design Pattern — centralizes creation of `SMS`,
    `WhatsApp`, and `CallTranscript` objects from a string type identifier.
    """

    _REGISTRY: dict[str, type[Message]] = {
        "sms": SMS,
        "whatsapp": WhatsApp,
        "wa": WhatsApp,
        "call": CallTranscript,
        "call_transcript": CallTranscript,
        "calltranscript": CallTranscript,
    }

    @classmethod
    def create(cls, msg_type: str, text: str, **kwargs: Any) -> Message:
        """Create and return a concrete `Message` instance for `msg_type`.

        Args:
            msg_type: Channel identifier ('sms', 'whatsapp', 'call_transcript').
            text: Raw message or transcript content.
            **kwargs: Optional subclass-specific keyword arguments.

        Raises:
            InvalidMessageTypeError: If `msg_type` is not recognized.
            EmptyMessageError: If `text` is empty or whitespace-only.
        """
        normalized = (msg_type or "").strip().lower().replace(" ", "_")
        message_cls = cls._REGISTRY.get(normalized)
        if message_cls is None:
            raise InvalidMessageTypeError(msg_type)
        return message_cls(text=text, **kwargs)
