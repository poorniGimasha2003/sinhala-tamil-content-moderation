# --- Step 1: Language Identification ---
from src.language_id.trigram_model import LanguageIdentifier
from data.processed.sample_words import SAMPLE_TRAINING_DATA

identifier = LanguageIdentifier(SAMPLE_TRAINING_DATA)

test_words = ["mama", "office", "vanakkam", "traffic"]
for word in test_words:
    lang, scores = identifier.predict(word)
    print(word, "->", lang)

sentence = "mama office ta yanawa traffic ekak thiyanawa"
result = identifier.predict_sentence(sentence)
for word, lang in result:
    print(word, "->", lang)

# --- Step 2: TF-IDF Feature Extraction ---
from src.classification.feature_extraction import build_tfidf_features

toy_comments = [
    "you are stupid",
    "you are nice",
    "have a nice day",
    "stupid stupid stupid"
]

vectorizer, tfidf_matrix = build_tfidf_features(toy_comments)

print("\nVocabulary:", vectorizer.get_feature_names_out())
print("\nTF-IDF matrix (rows=comments, columns=words):")
print(tfidf_matrix.toarray())

# --- Step 3: Naive Bayes Classifier (toy example) ---
from src.classification.classifier import OffensiveTextClassifier

toy_train_comments = [
    "you are so stupid", "get lost idiot", "shut up moron",
    "have a nice day", "thanks so much", "you are very kind",
    "this is stupid", "you are an idiot", "great job today",
    "have a wonderful morning"
]
toy_train_labels = [
    "offensive", "offensive", "offensive",
    "clean", "clean", "clean",
    "offensive", "offensive", "clean", "clean"
]

toy_classifier = OffensiveTextClassifier()
toy_classifier.train(toy_train_comments, toy_train_labels)

toy_test_comments = ["you are stupid", "have a great day", "shut up"]
for comment in toy_test_comments:
    label, probs = toy_classifier.predict(comment)
    print(f"'{comment}' -> {label} | probabilities: {probs}")

# --- Step 3b: Load real dataset ---
from src.classification.data_loader import load_hate_speech_data

comments, labels = load_hate_speech_data("data/raw/sentence-level-annotation.csv")

print("\nTotal comments loaded:", len(comments))
print("First comment:", comments[0])
print("First label:", labels[0])
print("Label distribution:", {l: labels.count(l) for l in set(labels)})

# --- Step 3c: Train/test split on real data ---
from src.classification.data_loader import split_data

train_comments, test_comments, train_labels, test_labels = split_data(comments, labels)

print("\nTraining set size:", len(train_comments))
print("Test set size:", len(test_comments))
print("Training label distribution:", {l: train_labels.count(l) for l in set(train_labels)})
print("Test label distribution:", {l: test_labels.count(l) for l in set(test_labels)})

# --- Step 3d: Train classifier on REAL data ---
real_classifier = OffensiveTextClassifier()
real_classifier.train(train_comments, train_labels)

print("\n--- Predictions on real test comments ---")
for comment, true_label in zip(test_comments[:5], test_labels[:5]):
    predicted_label, probs = real_classifier.predict(comment)
    print(f"Comment: {comment}")
    print(f"  True: {true_label} | Predicted: {predicted_label} | Probabilities: {probs}\n")

from src.classification.evaluation import evaluate_classifier

predictions = evaluate_classifier(real_classifier, test_comments, test_labels)

# --- Step 4: Balanced classifier (this is our production model going forward) ---
balanced_classifier = OffensiveTextClassifier()
balanced_classifier.train(train_comments, train_labels)  # balancing is on by default

print("\n--- Evaluation WITH class balancing ---")
predictions = evaluate_classifier(balanced_classifier, test_comments, test_labels)


# --- Transformer comparison (PAUSED — see README for findings) ---
# Fine-tuned XLM-RoBERTa was tested but showed the same class-imbalance
# failure as the unbalanced Naive Bayes baseline (0% recall on offensive).
# Needs class-weighted training to be a fair comparison — noted as future work.
#
# from src.classification.transformer_classifier import train_transformer_classifier
#
# subset_size = 2000
# small_train_comments = train_comments[:subset_size]
# small_train_labels = train_labels[:subset_size]
# small_test_comments = test_comments[:500]
# small_test_labels = test_labels[:500]
#
# print("\n--- Fine-tuning XLM-RoBERTa (this will take a while) ---")
# trainer, tokenizer = train_transformer_classifier(
#     small_train_comments, small_train_labels,
#     small_test_comments, small_test_labels
# )
#
# from src.classification.transformer_classifier import predict_with_transformer
# from sklearn.metrics import classification_report
#
# transformer_predictions = predict_with_transformer(trainer, tokenizer, small_test_comments)
#
# print("\n--- Transformer Evaluation ---")
# print(classification_report(small_test_labels, transformer_predictions))