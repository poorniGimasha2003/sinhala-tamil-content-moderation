# 🇱🇰 Code-Mixed Content Moderation Pipeline

![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![NLP](https://img.shields.io/badge/NLP-code--mixed%20text-orange)

An NLP pipeline for detecting offensive content in **Sinhala-Tamil-English code-mixed social media text** — the way people actually write online in Sri Lanka, mixing languages within the same sentence.

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
| 2 | Text preprocessing & feature extraction | ✅ Done |
| 3 | Classification model (Naive Bayes baseline; transformer attempted) | ✅ Done |
| 4 | Evaluation (precision/recall/F1, confusion matrix) | ✅ Done |
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

## ✅ Steps 2 & 3: Feature Extraction & Classification

Comments are converted to numeric TF-IDF vectors, then classified as `offensive` / `clean` using a Naive Bayes model trained on the real dataset (10,814 training comments).

```python
from src.classification.data_loader import load_hate_speech_data, split_data
from src.classification.classifier import OffensiveTextClassifier

comments, labels = load_hate_speech_data("data/raw/sentence-level-annotation.csv")
train_comments, test_comments, train_labels, test_labels = split_data(comments, labels)

classifier = OffensiveTextClassifier()
classifier.train(train_comments, train_labels)

label, probs = classifier.predict("some comment here")
```

**Note:** current results are unevaluated on the full test set — proper precision/recall/F1 evaluation is Step 4 (in progress). Given the dataset's class imbalance (~91% clean / 9% offensive), accuracy alone won't be a reliable metric here.
**Transformer comparison (attempted):** fine-tuned XLM-RoBERTa was tested but revealed the same class-imbalance failure mode as the unbalanced Naive Bayes baseline (0% recall on the offensive class) — confirming the imbalance issue is a property of the dataset, not the model. Class-weighted transformer training is noted as future work (`src/classification/transformer_classifier.py`).
---

## ✅ Step 4: Evaluation & Class Imbalance Fix

The dataset is heavily imbalanced (~91% clean / 9% offensive). An initial model trained without addressing this achieved 91% accuracy — but a closer look revealed it was catching only **1% of actual offensive comments** (recall = 0.01), simply defaulting to "clean" almost every time.

**Fix:** applied class-balanced sample weighting during training, forcing the model to actually learn from the minority (offensive) class.

| Metric (offensive class) | Before balancing | After balancing |
|---|---|---|
| Precision | 1.00 | 0.23 |
| Recall | 0.01 | 0.81 |
| F1-score | 0.02 | 0.36 |

**Takeaway:** overall accuracy dropped (91% → 74%), but this reflects a real tradeoff — the balanced model now catches 81% of offensive content instead of 1%, at the cost of more false positives. For a moderation system, missing real offensive content is the costlier failure, so this tradeoff is the right one.
## 📊 Dataset

Using the **[NLPC-UOM Sinhala-English Code-Mixed Dataset](https://huggingface.co/datasets/NLPC-UOM/Sinhala-English-Code-Mixed-Code-Switched-Dataset)** — 13,518 sentence-level annotated comments, originally labeled for sentiment, humor, and hate speech.

For this project, the 3-way hate speech label (`Not offensive` / `Abusive` / `Hate-Inducing`) is simplified into a binary `clean` / `offensive` label.

> ⚠️ Not included in this repo (see `.gitignore`) — download `sentence-level-annotation.csv` from the link above and place it in `data/raw/` to reproduce results.

**Planned for Tamil support:** [DravidianCodeMix](https://aclanthology.org/) — ~44,000 Tamil-English annotated YouTube comments (future work, see roadmap).

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

**Current:** Python · pandas · scikit-learn (TF-IDF, Naive Bayes)
**Planned:** Transformers (XLM-RoBERTa) · SHAP · FastAPI · Streamlit/Plotly

---

## 👤 Author

**Poorni Gimasha Pathirage**

🔗 [LinkedIn](https://www.linkedin.com/in/poornipathirage/) · [GitHub](https://github.com/poorniGimasha2003)

---

## 📄 License

MIT — see [LICENSE](LICENSE)