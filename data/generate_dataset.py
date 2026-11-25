"""Synthetic Indian scam & safe dataset generator for ScamShield.

Generates a balanced dataset of 1,800 messages (300 per class) across 6 classes:
    - Digital Arrest
    - Fake Courier
    - Fake KYC/Bank
    - Lottery/Prize
    - Job/Investment
    - Safe

Includes English, Hinglish (Hindi written in English letters), and mixed
templates with randomized Indian names, amounts, phone numbers, tracking IDs,
links, and realistic noise (typos, mixed casing, emojis).

Optionally incorporates benign ('ham') messages from the UCI SMS Spam
Collection if `data/SMSSpamCollection` exists locally, without depending on it.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SEED = 42
SAMPLES_PER_CLASS = 300
OUTPUT_PATH = Path("data") / "messages.csv"
UCI_PATH = Path("data") / "SMSSpamCollection"

NAMES = [
    "Rahul", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Rohan", "Pooja",
    "Suresh", "Neha", "Karan", "Kavita", "Rajesh", "Deepak", "Meena", "Arjun",
]
BANKS = ["SBI", "HDFC", "ICICI", "Axis Bank", "PNB", "Kotak", "Paytm", "PhonePe"]
COURIERS = ["FedEx", "DHL", "BlueDart", "India Post", "DTDC", "Delhivery"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune", "Jaipur", "Lucknow"]
SCAM_DOMAINS = [
    "bit.ly/kyc-verify-now",
    "tinyurl.com/claim-rs-now",
    "sbi-kyc-update.xyz/login",
    "fedex-customs-in.top/pay",
    "kbc-winner-claim.tk/form",
    "t.me/vip_earning_tasks",
    "cutt.ly/instant-loan-kyc",
    "hdfc-rekyc-alert.club/verify",
]
SAFE_DOMAINS = [
    "sbi.co.in",
    "irctc.co.in",
    "amazon.in",
    "flipkart.com",
    "uidai.gov.in",
]
EMOJIS_SCAM = ["🚨", "⚠️", "💰", "🎉", "🔥", "❗", "📲", "🎁", "💸"]
EMOJIS_SAFE = ["👍", "😊", "🙏", "✅", "📦", "☕"]


def _rand_phone(rng: random.Random) -> str:
    prefix = rng.choice(["+91-", "91", ""])
    first = rng.choice(["9", "8", "7", "6"])
    rest = "".join(str(rng.randint(0, 9)) for _ in range(9))
    return f"{prefix}{first}{rest}"


def _rand_amount(rng: random.Random, low: int = 500, high: int = 250000) -> str:
    val = rng.choice([499, 999, 1500, 2500, 5000, 12000, 25000, 50000, 150000, rng.randint(low, high)])
    fmt = rng.choice([f"Rs.{val:,}", f"₹{val:,}", f"INR {val}", f"Rs {val}"])
    return fmt


def _rand_id(rng: random.Random, prefix: str = "IND") -> str:
    return f"{prefix}{rng.randint(100000, 999999)}"


def _add_noise(text: str, rng: random.Random, is_scam: bool) -> str:
    """Inject realistic typos, casing variations, and occasional emojis."""
    if rng.random() < 0.25:
        # Simple realistic typos
        replacements = [
            ("immediately", rng.choice(["immediatly", "imediately"])),
            ("account", rng.choice(["acount", "accnt"])),
            ("verification", rng.choice(["verfication", "varification"])),
            ("congratulations", rng.choice(["congratulation", "congrts"])),
            ("parcel", rng.choice(["parsel", "parcal"])),
            ("please", rng.choice(["pls", "plz"])),
            ("update", rng.choice(["updat", "updt"])),
        ]
        for orig, typo in replacements:
            if orig in text.lower() and rng.random() < 0.45:
                text = text.replace(orig, typo).replace(orig.capitalize(), typo.capitalize())

    if rng.random() < 0.20:
        # Uppercase emphasis on a random word
        words = text.split()
        if len(words) > 4:
            idx = rng.randint(0, len(words) - 1)
            words[idx] = words[idx].upper()
            text = " ".join(words)

    if rng.random() < 0.30:
        emoji = rng.choice(EMOJIS_SCAM if is_scam else EMOJIS_SAFE)
        text = f"{emoji} {text}" if rng.random() < 0.5 else f"{text} {emoji}"

    return text


def _build_templates() -> dict[str, list[tuple[str, str, str]]]:
    """Return (template, message_type, language) tuples per class."""
    return {
        "Digital Arrest": [
            (
                "Caller: This is Inspector {name} from {city} Cyber Crime Branch. Your Aadhaar is linked to a money laundering case {id}. Stay on Skype video call for Digital Arrest verification or arrest warrant will be issued.",
                "call_transcript",
                "English",
            ),
            (
                "URGENT NOTICE from TRAI & CBI: Your mobile number {phone} will be disconnected in 2 hours due to illegal activities. Press 9 to speak to Enforcement Directorate officer or face non-bailable warrant.",
                "call_transcript",
                "English",
            ),
            (
                "Hello {name}, main Mumbai Police Narcotics Control Bureau se bol raha hu. Aapke naam pe illegal parcel pakda gaya hai. Video call on rakho aur disconnect mat karna, aap abhi Digital Arrest me hain.",
                "call_transcript",
                "Hinglish",
            ),
            (
                "CBI Cyber Cell Alert: Aapke Aadhaar card se 3 fake bank accounts khule hain. Turant secret video verification join karein warna 24 hours me giraftar ho jayenge. Call {phone}.",
                "whatsapp",
                "Mixed",
            ),
            (
                "Officer: Do not tell anyone or disconnect the call. You are under Digital Arrest by ED. Transfer {amount} to RBI safety verification account immediately to clear your name.",
                "call_transcript",
                "English",
            ),
            (
                "FINAL WARNING from Delhi Police Crime Branch: Case darj ho gaya hai against {name}. Apna bank balance verify karne ke liye abhi {amount} transfer karo warna police ghar aayegi.",
                "sms",
                "Hinglish",
            ),
            (
                "NCB Officer speaking: Drugs and fake passports found in your name. Stay on WhatsApp video call for statement recording. Do not inform family.",
                "call_transcript",
                "English",
            ),
            (
                "TRAI Official Alert: Aapka SIM 2 ghante me band hoga. Cyber crime case {id} solve karne ke liye CBI officer se turant video call par baat karein.",
                "sms",
                "Hinglish",
            ),
        ],
        "Fake Courier": [
            (
                "FedEx Alert: Your parcel {id} containing illegal items and MDMA drugs was seized by Mumbai Customs Department. Call {phone} immediately to avoid legal action.",
                "sms",
                "English",
            ),
            (
                "DHL Express: Delivery failed for package {id} due to incomplete address. Update your details and pay {amount} warehouse fee at {url}",
                "sms",
                "English",
            ),
            (
                "India Post: Aapka parcel {id} customs me ruka hua hai. Delivery reschedule karne ke liye {amount} customs clearance fee pay karein: {url}",
                "sms",
                "Hinglish",
            ),
            (
                "BlueDart Courier Notice: Hello {name}, aapke naam ka international parcel Mumbai airport par seized hua hai. Customs officer se baat karne ke liye {phone} dial karein.",
                "whatsapp",
                "Mixed",
            ),
            (
                "DTDC Tracking {id}: Your package is held at {city} hub. Pay {amount} redelivery charge within 24 hours via {url} otherwise parcel will be returned.",
                "sms",
                "English",
            ),
            (
                "Caller: Main FedEx {city} office se bol raha hu. Aapke Aadhaar se bheje parcel me illegal passport aur drugs mile hain. Customs clearance ke liye turant {amount} bhejo.",
                "call_transcript",
                "Hinglish",
            ),
            (
                "Delhivery Alert: Address wrong hone ke karan aapka parcel {id} hold par hai. Abhi link {url} par click karke address aur {amount} fee update karein.",
                "whatsapp",
                "Hinglish",
            ),
        ],
        "Fake KYC/Bank": [
            (
                "Dear {bank} customer, your KYC has expired! Your bank account will be blocked within 24 hours. Update PAN & KYC immediately at {url}",
                "sms",
                "English",
            ),
            (
                "{bank} Alert: Aapka khata aur netbanking aaj raat band ho jayega kyunki re-KYC pending hai. Turant KYC update karo: {url}",
                "sms",
                "Hinglish",
            ),
            (
                "Caller: Sir main {bank} KYC department se bol raha hu. Aapka UPI aur ATM card suspended ho raha hai. Verification ke liye 6 digit OTP aur ATM PIN batao.",
                "call_transcript",
                "Hinglish",
            ),
            (
                "URGENT {bank} Security: Suspicious login detected. Download AnyDesk or TeamViewer QuickSupport and share the 9-digit code with {phone} to protect {amount}.",
                "whatsapp",
                "English",
            ),
            (
                "Dear {name}, your {bank} PAN card link is incomplete. Account deactivated today if not verified. Click {url} and enter CVV & OTP.",
                "sms",
                "English",
            ),
            (
                "Aapka {bank} credit card reward points {amount} aaj expire ho rahe hain. Redeem karne ke liye OTP bhejo ya {url} par card details bharo.",
                "whatsapp",
                "Mixed",
            ),
            (
                "IMMEDIATE ACTION: Your SIM and {bank} UPI ID are suspended. Call {phone} and share MPIN to reactivate.",
                "sms",
                "English",
            ),
        ],
        "Lottery/Prize": [
            (
                "CONGRATULATIONS {name}! You have won {amount} in the KBC Kaun Banega Crorepati Lucky Draw! Call {phone} or visit {url} to claim your prize now.",
                "whatsapp",
                "English",
            ),
            (
                "Badhai ho! Aapka number {phone} KBC Jio bumper lottery me select hua hai. Aap {amount} jeet gaye hain! Inaam lene ke liye {amount} processing fee jama karein.",
                "whatsapp",
                "Hinglish",
            ),
            (
                "WINNER ALERT: Your mobile number won {amount} jackpot in India Mega Lottery {id}. Pay small registration charge via UPI to release prize money: {url}",
                "sms",
                "English",
            ),
            (
                "Namaste {name}, main KBC head office Mumbai se bol raha hu. Aapko {amount} ka bumper prize laga hai. Tax fee {amount} pehle transfer karo, turant paisa milega.",
                "call_transcript",
                "Hinglish",
            ),
            (
                "Amazon/Flipkart Anniversary Lucky Draw: You won an iPhone and {amount} cash voucher! Claim now before midnight at {url}",
                "whatsapp",
                "English",
            ),
            (
                "Jeet gaye aap {amount} ka cash inaam! Scan QR code to receive money or pay {amount} GST charge to {phone}.",
                "whatsapp",
                "Mixed",
            ),
        ],
        "Job/Investment": [
            (
                "Work from home part time job! Earn {amount} daily income just by liking YouTube videos and rating hotels. Join our Telegram mentor now: {url}",
                "whatsapp",
                "English",
            ),
            (
                "Ghar baithe roz {amount} kamaye! Bas Instagram aur YouTube like and subscribe task complete karo. Telegram HR se turant judein: {url}",
                "whatsapp",
                "Hinglish",
            ),
            (
                "VIP Crypto & Stock Trading Group: 100% profit guaranteed return! Double your money in 7 days. Invest {amount} today via {url}",
                "whatsapp",
                "English",
            ),
            (
                "Hello {name}, we saw your resume for Amazon part-time rating task. Daily earning {amount} guaranteed. Complete first prepaid task of {amount} to unlock bonus.",
                "sms",
                "English",
            ),
            (
                "Paisa double karne ka sunhera mauka! Hamare VIP trading group me {amount} lagao aur 48 hours me double profit pao. Contact {phone} on Telegram.",
                "whatsapp",
                "Hinglish",
            ),
            (
                "HR Manager: Aapko part time job ke liye select kiya gaya hai. Koi experience nahi chahiye, daily {amount} kamai. Registration fee {amount} pay karein.",
                "call_transcript",
                "Hinglish",
            ),
        ],
        "Safe": [
            (
                "Hi {name}, reaching {city} station by 6:30 PM. Let's meet for dinner near MG Road.",
                "whatsapp",
                "English",
            ),
            (
                "Your {bank} A/c XX4821 is debited by {amount} on 24-Nov via UPI. If not done by you, call official {bank} helpline.",
                "sms",
                "English",
            ),
            (
                "OTP for your {bank} netbanking login is 482910. Do NOT share this OTP with anyone, including bank employees.",
                "sms",
                "English",
            ),
            (
                "Bhai aaj shaam ko cricket khelne chaloge? 5 baje ground par milte hain.",
                "whatsapp",
                "Hinglish",
            ),
            (
                "Your Amazon order {id} has been delivered and handed to security. Thank you for shopping on {safe_url}.",
                "sms",
                "English",
            ),
            (
                "Reminder: Team standup meeting is rescheduled to 11:00 AM tomorrow. Please review the project slides before joining.",
                "whatsapp",
                "English",
            ),
            (
                "Mom: Ghar aate waqt 1 litre doodh aur sabzi le aana. Main khana bana rahi hu.",
                "whatsapp",
                "Hinglish",
            ),
            (
                "IRCTC Alert: Ticket PNR {id} confirmed for {city} Express. Coach B2, Seat 41. Have a safe journey!",
                "sms",
                "English",
            ),
            (
                "Caller: Hello {name}, I am calling from the dental clinic to confirm your appointment tomorrow at 4 PM. Let us know if you want to reschedule.",
                "call_transcript",
                "English",
            ),
            (
                "Kal college assignment submit karna hai, notes WhatsApp par bhej do pls.",
                "whatsapp",
                "Hinglish",
            ),
        ],
    }


def generate_dataset(
    output_path: Path = OUTPUT_PATH,
    samples_per_class: int = SAMPLES_PER_CLASS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate a balanced synthetic dataset and save to `output_path`."""
    rng = random.Random(seed)
    templates = _build_templates()
    rows: list[dict[str, str]] = []

    # Optional: load UCI SMS Spam Collection 'ham' messages if locally present
    uci_ham: list[str] = []
    if UCI_PATH.exists():
        for line in UCI_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[0].strip().lower() == "ham":
                uci_ham.append(parts[1].strip())
        rng.shuffle(uci_ham)

    for label, class_templates in templates.items():
        is_scam = label != "Safe"
        for i in range(samples_per_class):
            if label == "Safe" and uci_ham and i < min(60, len(uci_ham)):
                text = uci_ham[i]
                msg_type = "sms"
                lang = "English"
            else:
                tmpl, msg_type, lang = rng.choice(class_templates)
                text = tmpl.format(
                    name=rng.choice(NAMES),
                    bank=rng.choice(BANKS),
                    courier=rng.choice(COURIERS),
                    city=rng.choice(CITIES),
                    phone=_rand_phone(rng),
                    amount=_rand_amount(rng),
                    id=_rand_id(rng),
                    url=rng.choice(SCAM_DOMAINS),
                    safe_url=rng.choice(SAFE_DOMAINS),
                )
                text = _add_noise(text, rng, is_scam=is_scam)

            rows.append(
                {
                    "text": text,
                    "label": label,
                    "message_type": msg_type,
                    "language": lang,
                }
            )

    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    print(f"Generated {len(dataset)} messages saved to {OUTPUT_PATH}")
    print(dataset["label"].value_counts().to_string())
