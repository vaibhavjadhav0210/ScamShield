"""Comprehensive pytest suite for ScamShield OOP design and AI components."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from scamshield import (
    SMS,
    CallTranscript,
    DetectionResult,
    Detector,
    EmptyMessageError,
    Explainer,
    HistoryLog,
    HybridDetector,
    InvalidMessageTypeError,
    LLMExplainer,
    Message,
    MessageFactory,
    MLDetector,
    ModelNotTrainedError,
    RuleDetector,
    ScamShieldService,
    TemplateExplainer,
    TextCleaner,
    WhatsApp,
)


# ---------------------------------------------------------------------------
# 1. Models, Factory, Encapsulation, Polymorphism, Custom Exceptions
# ---------------------------------------------------------------------------


def test_message_factory_creates_sms() -> None:
    msg = MessageFactory.create("sms", "Your KYC is pending.")
    assert isinstance(msg, SMS)
    assert isinstance(msg, Message)
    assert msg.channel == "SMS"


def test_message_factory_creates_whatsapp() -> None:
    msg = MessageFactory.create("WhatsApp", "Join t.me/test now!", is_forwarded=True)
    assert isinstance(msg, WhatsApp)
    assert msg.is_forwarded is True


def test_message_factory_creates_call_transcript() -> None:
    msg = MessageFactory.create("call_transcript", "Caller: Stay on Skype video call.", duration_seconds=90)
    assert isinstance(msg, CallTranscript)
    assert msg.duration_seconds == 90


def test_message_factory_invalid_type_raises() -> None:
    with pytest.raises(InvalidMessageTypeError):
        MessageFactory.create("email", "Hello world")


@pytest.mark.parametrize("bad_text", ["", "   ", "\n\t"])
def test_empty_message_raises_custom_exception(bad_text: str) -> None:
    with pytest.raises(EmptyMessageError):
        SMS(bad_text)


def test_polymorphic_context_features() -> None:
    sms = SMS("VM-SBIBNK: Update KYC at http://bit.ly/abc")
    wa = WhatsApp("Join t.me/vip_tasks 🔥", is_forwarded=True)
    call = CallTranscript("Caller: Don't disconnect, stay on video call.")

    sms_ctx = sms.get_context_features()
    wa_ctx = wa.get_context_features()
    call_ctx = call.get_context_features()

    assert sms_ctx["has_dlt_header"] is True
    assert sms_ctx["has_url"] is True
    assert wa_ctx["has_group_or_telegram_link"] is True
    assert wa_ctx["has_emoji"] is True
    assert call_ctx["speaker_turns"] == 1
    assert call_ctx["mentions_video_or_ivr"] is True


def test_detection_result_is_immutable() -> None:
    res = DetectionResult(
        verdict="SCAM",
        scam_type="Fake KYC/Bank",
        risk_score=85,
        confidence=0.92,
        detector_name="RuleDetector",
    )
    with pytest.raises(FrozenInstanceError):
        res.risk_score = 10  # type: ignore[misc]


def test_detection_result_validation() -> None:
    with pytest.raises(ValueError):
        DetectionResult("MAYBE", "Safe", 10, 0.8, "RuleDetector")
    with pytest.raises(ValueError):
        DetectionResult("SCAM", "Fake Courier", 150, 0.8, "RuleDetector")


# ---------------------------------------------------------------------------
# 2. Preprocessing & Detectors
# ---------------------------------------------------------------------------


def test_text_cleaner_normalizes_urls_phones_money() -> None:
    cleaner = TextCleaner(preserve_placeholders=True)
    cleaned = cleaner.clean("Pay Rs.5,000 at https://bit.ly/xyz or call +91-9876543210!!!")
    assert "__money__" in cleaned
    assert "__url__" in cleaned
    assert "__phone__" in cleaned


def test_rule_detector_flags_digital_arrest() -> None:
    det = RuleDetector()
    msg = CallTranscript(
        "Caller: This is CBI Cyber Crime. You are under Digital Arrest. "
        "Stay on Skype video call and do not disconnect."
    )
    res = det.detect(msg)
    assert res.is_scam is True
    assert res.scam_type == "Digital Arrest"
    assert res.risk_score >= 70
    assert len(res.triggered_rules) >= 2


def test_rule_detector_flags_fake_kyc() -> None:
    det = RuleDetector()
    msg = SMS("Dear SBI user, KYC expired! Account blocked in 24 hours. Share OTP at bit.ly/kyc")
    res = det.detect(msg)
    assert res.is_scam is True
    assert res.scam_type == "Fake KYC/Bank"


def test_rule_detector_allows_safe_message() -> None:
    det = RuleDetector()
    msg = WhatsApp("Bhai shaam ko 6 baje coffee peene chalte hain.")
    res = det.detect(msg)
    assert res.verdict == "SAFE"
    assert res.scam_type == "Safe"
    assert res.risk_score < 35


def test_ml_detector_untrained_raises_model_not_trained() -> None:
    ml = MLDetector()
    with pytest.raises(ModelNotTrainedError):
        ml.detect(SMS("Test message"))


def test_ml_detector_fit_predict_save_load(tmp_path: Path) -> None:
    texts = [
        "CBI Digital Arrest warrant stay on Skype video call",
        "NCB police Aadhaar linked money laundering case",
        "FedEx parcel seized customs clearance fee pay now",
        "DHL delivery failed pay warehouse fee at link",
        "SBI KYC expired account blocked share OTP",
        "HDFC PAN update needed netbanking suspended",
        "Congratulations KBC lottery winner pay processing fee",
        "Lucky draw prize won claim cash now",
        "Work from home Telegram like subscribe daily income",
        "Guaranteed return crypto VIP group double money",
        "Hi Rahul let us meet for lunch tomorrow",
        "Team standup meeting at 11 AM in conference room",
    ] * 4
    labels = [
        "Digital Arrest", "Digital Arrest",
        "Fake Courier", "Fake Courier",
        "Fake KYC/Bank", "Fake KYC/Bank",
        "Lottery/Prize", "Lottery/Prize",
        "Job/Investment", "Job/Investment",
        "Safe", "Safe",
    ] * 4

    model_file = tmp_path / "ml.joblib"
    ml = MLDetector()
    ml.fit(texts, labels)
    ml.save(model_file)

    loaded = MLDetector(model_path=model_file)
    assert loaded.is_trained is True

    res = loaded.detect(SMS("SBI KYC expired account blocked share OTP immediately"))
    assert res.is_scam is True
    assert res.scam_type == "Fake KYC/Bank"
    assert len(res.top_features) > 0


def test_hybrid_detector_composes_ml_and_rules() -> None:
    texts = [
        "CBI Digital Arrest warrant Skype video call",
        "FedEx parcel seized customs fee",
        "SBI KYC expired account blocked OTP",
        "KBC lottery winner processing fee",
        "Work from home Telegram task daily income",
        "Meet me at the library at 5 pm",
    ] * 5
    labels = [
        "Digital Arrest",
        "Fake Courier",
        "Fake KYC/Bank",
        "Lottery/Prize",
        "Job/Investment",
        "Safe",
    ] * 5

    ml = MLDetector().fit(texts, labels)
    hybrid = HybridDetector(ml_detector=ml, rule_detector=RuleDetector())
    assert isinstance(hybrid, Detector)

    scam_res = hybrid.detect(SMS("FedEx parcel seized at customs, pay Rs.2000 immediately"))
    assert scam_res.is_scam is True
    assert scam_res.risk_score >= 45
    assert len(scam_res.triggered_rules) >= 1


# ---------------------------------------------------------------------------
# 3. Explainers, HistoryLog, and ScamShieldService (Strategy + Composition)
# ---------------------------------------------------------------------------


def test_template_explainer_outputs_reasons_and_tip() -> None:
    det = RuleDetector()
    msg = SMS("Congratulations! KBC lottery winner, pay processing fee now.")
    res = det.detect(msg)
    explainer: Explainer = TemplateExplainer()
    text = explainer.explain(msg, res)
    assert "SCAM" in text
    assert "Safety Tip:" in text


def test_llm_explainer_falls_back_when_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    explainer = LLMExplainer(api_key="")
    assert explainer.has_api_key is False

    msg = WhatsApp("Aapka KYC expired hai, turant OTP bhejo.")
    res = RuleDetector().detect(msg)
    explanation = explainer.explain(msg, res)
    assert "Safety Tip:" in explanation


def test_history_log_records_and_loads(tmp_path: Path) -> None:
    hist = HistoryLog(csv_path=tmp_path / "hist.csv")
    msg = SMS("Your parcel is held at customs.")
    res = RuleDetector().detect(msg)
    hist.record(msg, res)

    df = hist.load(limit=5)
    assert len(df) == 1
    assert df.iloc[0]["channel"] == "SMS"
    assert df.iloc[0]["verdict"] == res.verdict


def test_service_strategy_swap_and_end_to_end(tmp_path: Path) -> None:
    hist = HistoryLog(csv_path=tmp_path / "service_hist.csv")
    service = ScamShieldService(
        detector=RuleDetector(),
        explainer=TemplateExplainer(),
        history=hist,
    )
    res1, exp1 = service.analyze("whatsapp", "Ghar baithe daily income Telegram task join now!")
    assert res1.is_scam is True
    assert "Job/Investment" in res1.scam_type
    assert "Safety Tip:" in exp1

    # Strategy pattern: swap detector at runtime
    texts = ["CBI digital arrest", "Safe lunch meeting"] * 6
    labels = ["Digital Arrest", "Safe"] * 6
    ml = MLDetector().fit(texts, labels)
    service.set_detector(ml)

    res2, _ = service.analyze("call_transcript", "CBI officer calling for digital arrest")
    assert res2.detector_name == "MLDetector"
    assert len(hist.load()) == 2
