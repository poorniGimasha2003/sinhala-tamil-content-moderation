# 🇱🇰 Code-Mixed Content Moderation Pipeline

![Status](https://img.shields.io/badge/status-complete-brightgreen)
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


---

## 🗺️ Roadmap

| Step | Description | Status |
|------|-------------|--------|
| 1 | Language Identification (character-trigram model) | ✅ Done |
| 2 | Text preprocessing & feature extraction | ✅ Done |
| 3 | Classification model (Naive Bayes baseline; transformer attempted) | ✅ Done |
| 4 | Evaluation (precision/recall/F1, confusion matrix) | ✅ Done |
| 5 | Explainability (word-level Naive Bayes scores) | ✅ Done |
| 6 | REST API (FastAPI) | ✅ Done |
| 7 | Monitoring dashboard (Streamlit + Plotly) | ✅ Done |

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

---

## ✅ Step 5: Explainability

For each prediction, the top contributing words are surfaced by comparing each word's log-probability under `offensive` vs `clean` — directly readable from the trained Naive Bayes model's internal probabilities.

```python
from src.explainability.explain import explain_prediction

label, probs = balanced_classifier.predict("you are so stupid")
top_words = explain_prediction(balanced_classifier, "you are so stupid")
# [('stupid', 1.667), ('you', -0.721), ('are', -0.767), ('so', -1.271)]
```

**Known limitation:** this word-level view uses raw log-probabilities and doesn't account for TF-IDF weighting or the model's class prior — so it can diverge from the model's actual final decision (a genuine tradeoff of this lightweight approach vs. full tools like SHAP/LIME, noted as future work).

---

## ✅ Step 6: REST API

A FastAPI service exposes the full pipeline (classification + explainability) over HTTP.

```bash
python3 -m uvicorn src.api.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for interactive API testing, or:

```bash
curl -X POST 'http://127.0.0.1:8000/moderate' \
  -H 'Content-Type: application/json' \
  -d '{"text": "you are so stupid"}'
```

Returns the prediction, class probabilities, and top contributing words in one response.

---

## ✅ Step 7: Dashboard

An interactive Streamlit dashboard demonstrates the full pipeline: live comment moderation with explainability, and a visualization of the dataset's class imbalance.

```bash
python3 -m streamlit run dashboard/app.py
```

---

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

Python · pandas · scikit-learn (TF-IDF, Naive Bayes) · Transformers (XLM-RoBERTa) · FastAPI · Streamlit · Plotly

---

## 🔬 Limitations & Future Work

- **Class imbalance**: dataset is ~91% clean / 9% offensive. Addressed via class-weighted training (recall improved 0.01 → 0.81), but longer sentences with few strong signal words can still be misclassified due to dilution by common words.
- **Out-of-vocabulary words**: words not seen during training default to a 50/50 prediction with no explanation — a larger training set would reduce this.
- **Explainability accuracy**: the word-level explanation view uses raw log-probabilities and can diverge from the model's actual TF-IDF-weighted decision — a known tradeoff of this lightweight approach vs. full SHAP/LIME tooling.
- **Transformer upgrade**: XLM-RoBERTa fine-tuning was attempted but requires class-weighted training (not yet implemented) to be a fair comparison — noted as the clearest next step.
- **Tamil support**: language ID supports Tamil, but the classifier was only trained on the Sinhala-English dataset. DravidianCodeMix (Tamil-English) is identified as the dataset for extending this.

---

## 👤 Author

**Poorni Gimasha Pathirage**

🔗 [LinkedIn](https://www.linkedin.com/in/poornipathirage/) · [GitHub](https://github.com/poorniGimasha2003)

---

## 📄 License

MIT — see [LICENSE](LICENSE)