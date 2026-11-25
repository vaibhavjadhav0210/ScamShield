"""Rule-based scam detector using weighted Indian fraud heuristics.

OOP Concepts Demonstrated:
    - Inheritance: `RuleDetector` inherits from `Detector`.
    - Encapsulation: `_rules` and `_scam_threshold` are private attributes
      exposed via read-only properties; each `Rule` is an immutable dataclass.
    - Polymorphism: Implements `detect(message) -> DetectionResult` and
      incorporates `message.get_context_features()`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

from scamshield.detectors.base import Detector
from scamshield.models.message import DetectionResult, Message


@dataclass(frozen=True)
class Rule:
    """Immutable definition of a single fraud detection heuristic."""

    name: str
    category: str
    weight: int
    pattern: Pattern[str]
    reason: str


def _compile_default_rules() -> tuple[Rule, ...]:
    """Build the curated set of Indian English + Hinglish scam rules."""
    specs: list[tuple[str, str, int, str, str]] = [
        # --- Digital Arrest ---
        (
            "digital_arrest_agency",
            "Digital Arrest",
            38,
            r"\b(cbi|ncb|enforcement directorate|ed officer|cyber\s*crime|mumbai\s*police|delhi\s*police|trai|narcotics|crime\s*branch)\b",
            "Impersonates law enforcement or telecom authorities (CBI/NCB/TRAI/Police).",
        ),
        (
            "digital_arrest_threat",
            "Digital Arrest",
            42,
            r"\b(digital\s*arrest|arrest\s*warrant|non[- ]bailable|money\s*laundering|aadhaar\s*(?:misuse|linked)|giraftar|jail\s*ho\s*jayegi|case\s*darj)\b",
            "Threatens 'digital arrest', warrant, or fake criminal charges.",
        ),
        (
            "digital_arrest_video",
            "Digital Arrest",
            30,
            r"\b(skype|video\s*call\s*(?:statement|interrogation|verification)|stay\s*on\s*(?:the\s*)?line|don'?t\s*disconnect|disconnect\s*mat\s*karna|secret\s*verification)\b",
            "Pressures victim to stay on a video/voice call for 'verification'.",
        ),
        # --- Fake Courier ---
        (
            "fake_courier_brand",
            "Fake Courier",
            30,
            r"\b(fedex|dhl|bluedart|blue\s*dart|india\s*post|delhivery|dtdc|customs\s*department)\b",
            "Mentions a courier or customs department often spoofed in parcel scams.",
        ),
        (
            "fake_courier_contraband",
            "Fake Courier",
            40,
            r"\b(parcel\s*(?:seized|held|stuck|returned)|illegal\s*(?:items|drugs|passport)|mdma|contraband|customs\s*(?:duty|clearance|fee)|aapka\s*parcel\s*ruka)\b",
            "Claims a parcel was seized with illegal items or demands customs fees.",
        ),
        (
            "fake_courier_redelivery",
            "Fake Courier",
            28,
            r"\b(delivery\s*(?:failed|attempted|pending)|address\s*(?:incomplete|wrong)|reschedule\s*delivery|warehouse\s*fee)\b",
            "Uses fake failed-delivery urgency to lure clicks or small fee payments.",
        ),
        # --- Fake KYC/Bank ---
        (
            "fake_kyc_expiry",
            "Fake KYC/Bank",
            38,
            r"\b(kyc\s*(?:expired|pending|suspended|update|verify|band)|pan\s*(?:update|link|blocked)|re[- ]?kyc|kyc\s*karo|kyc\s*karwaye)\b",
            "Claims your bank/wallet KYC or PAN has expired and needs immediate update.",
        ),
        (
            "fake_bank_suspension",
            "Fake KYC/Bank",
            35,
            r"\b(account\s*(?:blocked|suspended|frozen|deactivated|closed)|khata\s*band|sim\s*(?:block|band)|upi\s*(?:blocked|suspended)|netbanking\s*blocked)\b",
            "Threatens imminent bank account, UPI, or SIM suspension.",
        ),
        (
            "credential_harvesting",
            "Fake KYC/Bank",
            38,
            r"\b(share\s*(?:your\s*)?otp|otp\s*(?:bhejo|batao|dena)|cvv|atm\s*pin|mpin|upi\s*pin|card\s*details|anydesk|teamviewer|quicksupport)\b",
            "Asks for OTP, PIN, CVV, or screen-sharing apps (AnyDesk/TeamViewer).",
        ),
        # --- Lottery/Prize ---
        (
            "lottery_claim",
            "Lottery/Prize",
            40,
            r"\b(congratulations|kbc|kaun\s*banega\s*crorepati|lucky\s*draw|lottery\s*(?:winner|jeete)|jackpot|bumper\s*prize|jeet\s*gaye|inaam|lakh\s*won)\b",
            "Promises an unsolicited lottery, KBC, or lucky draw prize.",
        ),
        (
            "lottery_processing_fee",
            "Lottery/Prize",
            35,
            r"\b(processing\s*fee|registration\s*charge|tax\s*fee\s*to\s*claim|claim\s*(?:your\s*)?(?:prize|reward|cash)|gift\s*voucher\s*won)\b",
            "Demands a processing fee or tax upfront to release a fake prize.",
        ),
        # --- Job/Investment ---
        (
            "job_wfh_task",
            "Job/Investment",
            36,
            r"\b(work\s*from\s*home|part[- ]?time\s*job|ghar\s*baithe|daily\s*(?:income|earning|kamai)|like\s*(?:and|&)\s*subscribe|youtube\s*task|telegram\s*(?:task|mentor|hr)|rating\s*task)\b",
            "Offers unrealistic work-from-home or Telegram 'like/rating' task income.",
        ),
        (
            "investment_guaranteed_return",
            "Job/Investment",
            38,
            r"\b(guaranteed\s*(?:return|profit)|double\s*(?:your\s*)?money|paisa\s*double|100%\s*profit|crypto\s*signal|vip\s*trading\s*group|prepaid\s*task)\b",
            "Promises guaranteed/doubled investment returns or prepaid task profits.",
        ),
        # --- Cross-cutting Scam Signals ---
        (
            "urgency_pressure",
            "General",
            18,
            r"\b(urgent|immediately|within\s*\d+\s*(?:hours?|hrs?|minutes?|mins?)|last\s*warning|final\s*notice|aaj\s*hi|turant|abhi\s*karo|jaldi\s*karo)\b",
            "Creates artificial urgency ('immediately', 'last warning', 'turant').",
        ),
        (
            "suspicious_link",
            "General",
            22,
            r"(?:bit\.ly|tinyurl\.com|cutt\.ly|rb\.gy|t\.me/|chat\.whatsapp\.com|https?://[^\s]*\.(?:xyz|top|tk|ml|ga|cf|club|info)\b|kyc[-_]?update|claim[-_]?prize)",
            "Contains a shortened, Telegram, or suspicious lookalike URL.",
        ),
        (
            "money_transfer_request",
            "General",
            22,
            r"\b(pay\s*(?:now|immediately|via\s*upi)|transfer\s*(?:rs|₹|inr|money)|send\s*(?:rs|₹|inr)|upi\s*id\s*:|bhej\s*do|scan\s*qr\s*to\s*receive)\b",
            "Requests direct money transfer, UPI payment, or scanning a QR code.",
        ),
    ]
    return tuple(
        Rule(
            name=name,
            category=category,
            weight=weight,
            pattern=re.compile(pattern, re.IGNORECASE),
            reason=reason,
        )
        for name, category, weight, pattern, reason in specs
    )


class RuleDetector(Detector):
    """Heuristic rule-based detector tailored for Indian scam patterns."""

    def __init__(self, scam_threshold: int = 35) -> None:
        if not 1 <= scam_threshold <= 100:
            raise ValueError("scam_threshold must be between 1 and 100.")
        self._rules: tuple[Rule, ...] = _compile_default_rules()
        self._scam_threshold: int = scam_threshold

    @property
    def rules(self) -> tuple[Rule, ...]:
        """Return the immutable tuple of configured detection rules."""
        return self._rules

    @property
    def scam_threshold(self) -> int:
        """Return the minimum risk score required to classify a message as SCAM."""
        return self._scam_threshold

    def detect(self, message: Message) -> DetectionResult:
        """Evaluate all heuristic rules and channel context on `message`."""
        text = message.text
        context = message.get_context_features()

        triggered: list[Rule] = [rule for rule in self._rules if rule.pattern.search(text)]
        raw_score = sum(rule.weight for rule in triggered)

        # Channel-specific contextual boosts (demonstrating polymorphic context usage)
        if context.get("mentions_video_or_ivr") and any(
            r.category == "Digital Arrest" for r in triggered
        ):
            raw_score += 10
        if context.get("has_group_or_telegram_link") and any(
            r.category == "Job/Investment" for r in triggered
        ):
            raw_score += 10

        risk_score = min(100, raw_score)

        # Aggregate score per specific scam category
        category_scores: dict[str, int] = {}
        for rule in triggered:
            if rule.category != "General":
                category_scores[rule.category] = category_scores.get(rule.category, 0) + rule.weight

        if risk_score >= self._scam_threshold and category_scores:
            scam_type = max(category_scores, key=lambda cat: category_scores[cat])
            verdict = "SCAM"
        elif risk_score >= self._scam_threshold:
            # General rules triggered without a specific category -> default to Fake KYC/Bank
            scam_type = "Fake KYC/Bank"
            verdict = "SCAM"
        else:
            scam_type = "Safe"
            verdict = "SAFE"

        confidence = (
            min(0.99, max(0.51, risk_score / 100.0))
            if verdict == "SCAM"
            else min(0.99, max(0.55, (100 - risk_score) / 100.0))
        )

        return DetectionResult(
            verdict=verdict,
            scam_type=scam_type,
            risk_score=risk_score,
            confidence=round(confidence, 4),
            detector_name=self.name,
            triggered_rules=tuple(r.name for r in triggered),
            rule_reasons=tuple(r.reason for r in triggered),
            top_features=(),
        )
