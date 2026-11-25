"""Text preprocessing utilities for ScamShield.

Normalizes noisy English and Hinglish SMS/WhatsApp/Call texts while preserving
high-signal fraud tokens (URLs, currency amounts, phone numbers, OTPs).
"""

from __future__ import annotations

import re


class TextCleaner:
    """Stateless/configurable text preprocessor for English and Hinglish messages.

    OOP Concept: Encapsulation of preprocessing configuration and regex pipelines
    behind a clean `clean()` interface.
    """

    _URL_RE = re.compile(
        r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(?:xyz|top|tk|ml|ga|cf|info|club|in|com|org)/\S*)",
        re.IGNORECASE,
    )
    _PHONE_RE = re.compile(r"(?:\+?91[\s-]?)?[6-9]\d{9}\b")
    _CURRENCY_RE = re.compile(r"(?:₹|rs\.?|inr)\s*[\d,]+(?:\.\d+)?", re.IGNORECASE)
    _WHITESPACE_RE = re.compile(r"\s+")
    _REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
    _NON_ALNUM_RE = re.compile(r"[^\w\s_]")

    def __init__(self, preserve_placeholders: bool = True) -> None:
        self._preserve_placeholders: bool = preserve_placeholders

    @property
    def preserve_placeholders(self) -> bool:
        """Whether URLs, phone numbers, and currency amounts are mapped to tokens."""
        return self._preserve_placeholders

    def clean(self, text: str) -> str:
        """Normalize raw message text for ML feature extraction.

        Args:
            text: Input string in English, Hinglish, or mixed script.

        Returns:
            Cleaned lowercase string with normalized whitespace and optional
            semantic placeholders (`__url__`, `__phone__`, `__money__`).
        """
        if not text:
            return ""

        cleaned = text.strip()
        if self._preserve_placeholders:
            cleaned = self._URL_RE.sub(" __url__ ", cleaned)
            cleaned = self._PHONE_RE.sub(" __phone__ ", cleaned)
            cleaned = self._CURRENCY_RE.sub(" __money__ ", cleaned)

        cleaned = cleaned.lower()
        # Collapse exaggerated character repetitions like "urggeeenttt" -> "urgeent"
        cleaned = self._REPEATED_CHAR_RE.sub(r"\1\1", cleaned)
        # Keep word characters, spaces, and underscores (for placeholders)
        cleaned = self._NON_ALNUM_RE.sub(" ", cleaned)
        cleaned = self._WHITESPACE_RE.sub(" ", cleaned).strip()
        return cleaned
