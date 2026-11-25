# 🛡️ ScamShield — Indian Scam Message & Call Transcript Detector

ScamShield is an OOP + AI project that detects Indian SMS, WhatsApp, and phone/video call scams in **English, Hinglish, or mixed text**. It classifies messages into 6 categories (**Digital Arrest**, **Fake Courier**, **Fake KYC/Bank**, **Lottery/Prize**, **Job/Investment**, or **Safe**), outputs a **0–100 risk score**, and explains **why** in simple English.

---

## Key Features
- **Hybrid AI + Rules Engine**: Combines a TF-IDF (word + character n-grams) + LogisticRegression classifier (`MLDetector`) with curated Indian fraud heuristics (`RuleDetector`) via `HybridDetector` (`0.6 * ML + 0.4 * Rules`).
- **Strong OOP Architecture**: Demonstrates Abstraction (`ABC`), Inheritance, Polymorphism, Encapsulation (`frozen=True` `DetectionResult`, private attributes + `@property`), Composition, Strategy & Factory patterns, and Custom Exceptions. See [`docs/oop_concepts.md`](docs/oop_concepts.md) and [`docs/uml_class_diagram.md`](docs/uml_class_diagram.md).
- **Offline-First Explanations**: `TemplateExplainer` works 100% offline. If `ANTHROPIC_API_KEY` is set in the environment, `LLMExplainer` uses Anthropic Claude and falls back to `TemplateExplainer` on any error.
- **Interactive Streamlit UI**: Includes sample Indian scam messages, risk meter, strategy switcher, safety tips, and CSV check history (`data/history.csv`).

---

## Project Structure
```text
ScamShield/
├── app.py                      # Streamlit UI
├── train.py                    # 80/20 stratified split + 5-fold CV + model export
├── evaluate.py                 # Rule vs ML vs Hybrid benchmark + charts
├── requirements.txt            # Pinned dependencies
├── data/
│   ├── generate_dataset.py     # Synthetic English/Hinglish dataset generator (1,800 rows)
│   └── messages.csv            # Generated labeled dataset
├── scamshield/
│   ├── exceptions.py           # EmptyMessageError, ModelNotTrainedError, InvalidMessageTypeError
│   ├── preprocess.py           # TextCleaner
│   ├── history.py              # HistoryLog (CSV persistence)
│   ├── service.py              # ScamShieldService (Facade + Strategy)
│   ├── models/message.py       # Message, SMS, WhatsApp, CallTranscript, MessageFactory, DetectionResult
│   ├── detectors/              # Detector (ABC), RuleDetector, MLDetector, HybridDetector
│   └── explain/explainer.py    # Explainer (ABC), TemplateExplainer, LLMExplainer
├── tests/                      # Pytest suite (20+ tests)
├── results/                    # metrics.json, confusion_matrix.png, comparison.png, misclassified.txt
└── docs/                       # uml_class_diagram.md, oop_concepts.md
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python data/generate_dataset.py
```

### 3. Train the ML model (80/20 stratified split + 5-fold CV)
```bash
python train.py
```

### 4. Evaluate Rule vs ML vs Hybrid detectors
```bash
python evaluate.py
```
Outputs are saved to `results/` (`metrics.json`, `confusion_matrix.png`, `comparison.png`, `misclassified.txt`).

### 5. Run unit & integration tests
```bash
pytest
```

### 6. Launch the Streamlit app
```bash
streamlit run app.py
```
*(Optional)* To enable LLM-generated explanations, set `ANTHROPIC_API_KEY` before launching Streamlit. Without it, ScamShield runs completely offline using `TemplateExplainer`.

---

## Disclaimer & Dataset Note
> **Important:** The training and evaluation dataset (`data/messages.csv`) is **synthetically generated** using varied English and Hinglish templates with randomized names, amounts, phone numbers, URLs, and noise. Consequently, metrics on this synthetic test split will be higher than on unseen real-world messages. ScamShield is an academic tool and can make mistakes—always verify suspicious messages through official channels or call **1930** (India National Cybercrime Helpline).
