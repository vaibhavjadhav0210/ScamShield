"""Explanation generators for ScamShield detection results.

OOP Concepts Demonstrated:
    - Abstraction: `Explainer` is an abstract base class (`abc.ABC`) defining
      the `explain(message, result) -> str` contract.
    - Inheritance: `TemplateExplainer` and `LLMExplainer` inherit from `Explainer`.
    - Polymorphism: `ScamShieldService` calls `explainer.explain(message, result)`
      without knowing whether offline templates or an LLM generated the text.
    - Composition & Fallback: `LLMExplainer` composes a `TemplateExplainer` as its
      reliable offline fallback whenever `ANTHROPIC_API_KEY` is missing or an API
      call fails.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import requests

from scamshield.models.message import DetectionResult, Message

_SAFETY_TIPS: dict[str, str] = {
    "Digital Arrest": (
        "Safety Tip: Indian police, CBI, NCB, or TRAI NEVER conduct 'digital arrests' "
        "or interrogations over Skype/WhatsApp video calls. Disconnect immediately and "
        "verify by calling 1930 (National Cybercrime Helpline)."
    ),
    "Fake Courier": (
        "Safety Tip: Do not click unknown tracking links or pay 'customs clearance' fees "
        "for parcels you never ordered. Verify directly on the official courier website."
    ),
    "Fake KYC/Bank": (
        "Safety Tip: Banks NEVER ask for OTP, PIN, CVV, or KYC updates via SMS/WhatsApp "
        "links. Never install AnyDesk/TeamViewer on a stranger's request."
    ),
    "Lottery/Prize": (
        "Safety Tip: Legitimate lotteries or KBC prizes NEVER ask you to pay a "
        "'processing fee' or tax upfront to claim winnings. Ignore and block the sender."
    ),
    "Job/Investment": (
        "Safety Tip: Beware of Telegram 'like/subscribe' tasks or 'guaranteed return' "
        "crypto/trading groups. Never pay a registration or prepaid task deposit."
    ),
    "Safe": (
        "Safety Tip: This message appears safe, but always stay cautious before "
        "sharing personal details, OTPs, or clicking unexpected links."
    ),
}


class Explainer(ABC):
    """Abstract base class for generating human-readable scam explanations."""

    @abstractmethod
    def explain(self, message: Message, result: DetectionResult) -> str:
        """Generate a plain-English explanation of `result` for `message`."""


class TemplateExplainer(Explainer):
    """Offline, deterministic explainer using triggered rules, ML words, and tips."""

    def explain(self, message: Message, result: DetectionResult) -> str:
        """Build a clear, simple-English explanation without external APIs."""
        lines: list[str] = []
        context = message.get_context_features()

        if result.is_scam:
            lines.append(
                f"This {message.channel} message is flagged as **{result.verdict}** "
                f"(Category: **{result.scam_type}**, Risk Score: **{result.risk_score}/100**)."
            )
            if result.rule_reasons:
                lines.append("Why it looks suspicious:")
                for reason in result.rule_reasons:
                    lines.append(f"- {reason}")
            if result.top_features:
                keywords = ", ".join(f"'{w}'" for w in result.top_features[:5])
                lines.append(f"Key AI-detected indicator phrases: {keywords}.")
            if context.get("mentions_video_or_ivr"):
                lines.append("- Mentions video call / staying on the line, a common pressure tactic.")
            if context.get("has_group_or_telegram_link"):
                lines.append("- Contains a WhatsApp/Telegram redirect link often used in task scams.")
        else:
            lines.append(
                f"This {message.channel} message appears **SAFE** "
                f"(Risk Score: **{result.risk_score}/100**)."
            )
            lines.append(
                "No high-risk scam patterns, fake urgency cues, or suspicious credential "
                "requests were detected."
            )
            if result.top_features:
                keywords = ", ".join(f"'{w}'" for w in result.top_features[:4])
                lines.append(f"Normal/benign language cues observed: {keywords}.")

        tip = _SAFETY_TIPS.get(result.scam_type, _SAFETY_TIPS["Safe"])
        lines.append("")
        lines.append(tip)
        return "\n".join(lines)


class LLMExplainer(Explainer):
    """Optional Anthropic-powered explainer with automatic offline fallback.

    Reads `ANTHROPIC_API_KEY` from the environment. If the key is absent or any
    API/network error occurs, it transparently falls back to `TemplateExplainer`.
    """

    _API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-haiku-latest",
        timeout_seconds: float = 8.0,
    ) -> None:
        self._api_key: str | None = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self._model: str = model
        self._timeout: float = timeout_seconds
        # Composition: reliable offline fallback
        self._fallback: TemplateExplainer = TemplateExplainer()

    @property
    def has_api_key(self) -> bool:
        """Return True when an Anthropic API key is configured."""
        return bool(self._api_key and self._api_key.strip())

    def explain(self, message: Message, result: DetectionResult) -> str:
        """Explain via Anthropic API if configured, otherwise fall back offline."""
        if not self.has_api_key:
            return self._fallback.explain(message, result)

        prompt = (
            "You are ScamShield, a consumer safety assistant in India. "
            "Explain in 3-4 short, simple English bullet points why this message was "
            f"classified as {result.verdict} ({result.scam_type}, risk {result.risk_score}/100). "
            "End with one practical safety tip.\n\n"
            f"Channel: {message.channel}\n"
            f"Triggered rules: {', '.join(result.triggered_rules) or 'None'}\n"
            f"ML keywords: {', '.join(result.top_features) or 'None'}\n"
            f"Message: {message.text[:600]}"
        )

        try:
            response = requests.post(
                self._API_URL,
                headers={
                    "x-api-key": self._api_key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": 220,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            content_blocks = data.get("content", [])
            text_parts = [
                block.get("text", "")
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            explanation = "\n".join(p.strip() for p in text_parts if p.strip())
            return explanation if explanation else self._fallback.explain(message, result)
        except Exception:
            # Graceful degradation: never break the app if API fails
            return self._fallback.explain(message, result)
