"""Streamlit web interface for ScamShield."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from scamshield.detectors import HybridDetector, MLDetector, RuleDetector
from scamshield.detectors.ml import DEFAULT_MODEL_PATH
from scamshield.exceptions import EmptyMessageError, ModelNotTrainedError
from scamshield.explain import LLMExplainer, TemplateExplainer
from scamshield.history import HistoryLog
from scamshield.service import ScamShieldService

SAMPLE_MESSAGES: dict[str, tuple[str, str]] = {
    "1. Digital Arrest (Call Transcript)": (
        "call_transcript",
        "Caller: This is Inspector Verma from Mumbai Cyber Crime Branch. Your Aadhaar "
        "is linked to a money laundering parcel. Stay on Skype video call for Digital Arrest "
        "verification or non-bailable warrant will be issued. Do not disconnect!",
    ),
    "2. Fake Courier (SMS)": (
        "sms",
        "FedEx Alert: Your parcel IND849201 containing illegal items was seized by Mumbai "
        "Customs. Call +91-9876543210 immediately or pay Rs.1,500 at fedex-customs-in.top/pay",
    ),
    "3. Fake KYC / Bank (Hinglish SMS)": (
        "sms",
        "SBI Alert: Aapka khata aur netbanking aaj raat band ho jayega kyunki KYC expired hai. "
        "Turant PAN & KYC update karo: bit.ly/kyc-verify-now warna account blocked hoga.",
    ),
    "4. Lottery / KBC Prize (WhatsApp)": (
        "whatsapp",
        "CONGRATULATIONS! Aapka number KBC Kaun Banega Crorepati Lucky Draw me Rs.25,00,000 "
        "jeet gaya hai! Claim now at kbc-winner-claim.tk/form and pay Rs.2,500 processing fee.",
    ),
    "5. Job / Investment Task (WhatsApp)": (
        "whatsapp",
        "Ghar baithe roz Rs.5,000 kamaye! Work from home part time job: bas YouTube like and "
        "subscribe task complete karo. Join our Telegram mentor: t.me/vip_earning_tasks",
    ),
}


@st.cache_resource
def _load_ml_detector() -> MLDetector:
    """Load the pre-trained MLDetector artifact once per Streamlit session."""
    if DEFAULT_MODEL_PATH.exists():
        return MLDetector(model_path=DEFAULT_MODEL_PATH)
    return MLDetector()


def main() -> None:
    st.set_page_config(page_title="ScamShield India", page_icon="🛡️", layout="wide")

    st.title("🛡️ ScamShield — Indian Scam Message & Call Detector")
    st.caption(
        "Detects **Digital Arrest**, **Fake Courier**, **Fake KYC/Bank**, **Lottery/Prize**, "
        "and **Job/Investment** scams across English and Hinglish messages."
    )

    st.info(
        "**Academic Disclaimer:** ScamShield is an academic OOP + AI project trained on a "
        "synthetic dataset. It can make mistakes—always verify suspicious messages via official "
        "channels or call **1930** (India National Cybercrime Helpline)."
    )

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    if "msg_type" not in st.session_state:
        st.session_state["msg_type"] = "sms"

    history_log = HistoryLog()

    # --- Sidebar ---
    with st.sidebar:
        st.header("📌 Sample Scam Messages")
        st.write("Click any sample to load it into the checker:")
        for label, (sample_type, sample_text) in SAMPLE_MESSAGES.items():
            if st.button(label, use_container_width=True):
                st.session_state["input_text"] = sample_text
                st.session_state["msg_type"] = sample_type

        st.divider()
        st.header("🧠 Detection Strategy")
        detector_choice = st.selectbox(
            "Select Detector",
            options=["Hybrid (ML + Rules)", "ML Only (TF-IDF + LogReg)", "Rules Only"],
            index=0,
        )
        llm_status = (
            "Anthropic LLM + Offline Fallback"
            if os.environ.get("ANTHROPIC_API_KEY")
            else "Offline TemplateExplainer (No API key needed)"
        )
        st.caption(f"Explainer: `{llm_status}`")

        st.divider()
        st.header("🛡️ Safety Tips")
        st.markdown(
            "- **Never share OTP, PIN, or CVV** with anyone.\n"
            "- **No 'Digital Arrest' exists** in Indian law—police never interrogate on Skype/WhatsApp.\n"
            "- **Don't click shortened KYC/parcel links** (`bit.ly`, `.xyz`, `.top`).\n"
            "- **Report cyber fraud immediately:** Call **1930** or visit "
            "[cybercrime.gov.in](https://cybercrime.gov.in)."
        )

    # --- Main Input Area ---
    col1, col2 = st.columns([3, 1])
    with col2:
        type_options = {"SMS": "sms", "WhatsApp": "whatsapp", "Call Transcript": "call_transcript"}
        reverse_map = {v: k for k, v in type_options.items()}
        selected_label = st.selectbox(
            "Message Type",
            options=list(type_options.keys()),
            index=list(type_options.keys()).index(
                reverse_map.get(st.session_state["msg_type"], "SMS")
            ),
        )
        st.session_state["msg_type"] = type_options[selected_label]

    with col1:
        user_text = st.text_area(
            "Paste SMS, WhatsApp message, or Call Transcript here:",
            value=st.session_state["input_text"],
            height=150,
            placeholder="e.g. Dear customer, your SBI KYC has expired. Update at bit.ly/...",
        )
        st.session_state["input_text"] = user_text

    if st.button("🔍 Check Message", type="primary"):
        ml_det = _load_ml_detector()
        rule_det = RuleDetector()

        if detector_choice.startswith("Rules"):
            active_detector = rule_det
        elif detector_choice.startswith("ML"):
            active_detector = ml_det
        else:
            active_detector = HybridDetector(ml_detector=ml_det, rule_detector=rule_det)

        explainer = LLMExplainer() if os.environ.get("ANTHROPIC_API_KEY") else TemplateExplainer()
        service = ScamShieldService(
            detector=active_detector,
            explainer=explainer,
            history=history_log,
        )

        try:
            result, explanation = service.analyze(
                msg_type=st.session_state["msg_type"],
                text=user_text,
            )

            st.divider()
            if result.is_scam:
                st.error(f"### 🚨 VERDICT: SCAM DETECTED — {result.scam_type}")
            else:
                st.success("### ✅ VERDICT: SAFE")

            m1, m2, m3 = st.columns(3)
            m1.metric("Risk Score", f"{result.risk_score} / 100")
            m2.metric("Category", result.scam_type)
            m3.metric("Confidence", f"{int(round(result.confidence * 100))}%")

            st.write("**Risk Meter**")
            st.progress(result.risk_score / 100.0)

            st.subheader("Why?")
            st.markdown(explanation)

        except EmptyMessageError:
            st.warning("Please paste a non-empty message or transcript before clicking Check.")
        except ModelNotTrainedError:
            st.error(
                "ML model artifact not found. Run `python data/generate_dataset.py` and "
                "`python train.py` first, or switch to 'Rules Only'."
            )

    # --- Recent History Table ---
    st.divider()
    st.subheader("📜 Recent Check History")
    hist_df = history_log.load(limit=15)
    if hist_df.empty:
        st.caption("No checks recorded yet. Analyze a message above to see history.")
    else:
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
