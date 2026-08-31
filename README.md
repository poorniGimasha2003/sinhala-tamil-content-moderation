# 🇱🇰 Code-Mixed Content Moderation Pipeline

![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![NLP](https://img.shields.io/badge/NLP-code--mixed%20text-orange)

An NLP pipeline for detecting offensive content in **Sinhala-Tamil-English code-mixed social media text** — the way people actually write online in Sri Lanka, mixing languages within the same sentence.

> 🚧 **Work in progress** — built step-by-step as a learning-driven portfolio project. Progress tracked below.

---

## 💡 Why this project?

Most content moderation systems are built for single-language, "clean" text. But real comments look like this:

> *"mama office ta yanawa traffic ekak thiyanawa"*

Sinhala, Tamil, and English mixed together in one sentence — a genuinely hard, underexplored NLP problem. Standard moderation tools often miss offensive content simply because they can't parse mixed-language text properly.

This project builds a small end-to-end system — similar in shape to what a real **Trust & Safety** engineering team would build — from scratch.

---

## 🏗️ Pipeline Architecture

```
📥 Raw comment
     │
     ▼
1️⃣  Language ID          (per-word: Sinhala / Tamil / English)
     │
     ▼
2️⃣  Preprocessing
     │
     ▼
3️⃣  Classification        (offensive / clean)
     │
     ▼
4️⃣  Explainability        (which words triggered the flag)
     │
     ▼
5️⃣  REST API  →  📊 Dashboard
```


---

## 🗺️ Roadmap

| Step | Description | Status |
|------|-------------|--------|
| 1 | Language Identification (character-trigram model) | ✅ Done |
| 2 | Text preprocessing & feature extraction | ⬜ Planned |
| 3 | Classification model (classical ML → transformer) | ⬜ Planned |
| 4 | Evaluation (precision/recall/F1, confusion matrix) | ⬜ Planned |
| 5 | Explainability (LIME/SHAP) | ⬜ Planned |
| 6 | REST API (FastAPI) | ⬜ Planned |
| 7 | Monitoring dashboard (visualizations) | ⬜ Planned |

---

## ✅ Step 1: Language Identification

Detects the language of each word using **character-trigram probability models** (Laplace-smoothed) — the same n-gram approach used for word-level language models, applied at the character level instead.

```python
from src.language_id.trigram_model import LanguageIdentifier
from data.processed.sample_words import SAMPLE_TRAINING_DATA

identifier = LanguageIdentifier(SAMPLE_TRAINING_DATA)
result = identifier.predict_sentence("mama office ta yanawa traffic ekak")
# [('mama', 'sinhala'), ('office', 'english'), ('ta', 'sinhala'), ...]
```

**How it works:** each word is broken into overlapping 3-letter chunks (trigrams). Each language has its own trigram frequency table built from training data. A new word is scored against all three tables using log-probability, and the highest-scoring language wins.

---

## 📊 Datasets (planned for later steps)

- **[NLPC-UOM Sinhala-English Code-Mixed Dataset](https://huggingface.co/)** — 10,000 manually annotated comments (sentiment, humor, hate speech, language ID)
- **[DravidianCodeMix](https://aclanthology.org/)** — ~44,000 Tamil-English annotated YouTube comments (offensive language)

---

## ⚙️ Setup

```bash
git clone https://github.com/poorniGimasha2003/sinhala-tamil-content-moderation.git
cd sinhala-tamil-content-moderation
pip install -r requirements.txt
python3 test_run.py
```

---

## 🛠️ Tech Stack

**Current:** Python
**Planned:** scikit-learn · Transformers (XLM-RoBERTa) · SHAP · FastAPI · Streamlit/Plotly

---

## 👤 Author

**Poorni Gimasha Pathirage** 

🔗 [LinkedIn](https://www.linkedin.com/in/poornipathirage/) · [GitHub](https://github.com/poorniGimasha2003)

---

## 📄 License

MIT — see [LICENSE](LICENSE)