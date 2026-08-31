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

# --- Step 2: TF-IDF Feature Extraction (new code) ---
from src.classification.feature_extraction import build_tfidf_features

comments = [
    "you are stupid",
    "you are nice",
    "have a nice day",
    "stupid stupid stupid"
]

vectorizer, tfidf_matrix = build_tfidf_features(comments)

print("\nVocabulary:", vectorizer.get_feature_names_out())
print("\nTF-IDF matrix (rows=comments, columns=words):")
print(tfidf_matrix.toarray())

# --- Step 3:

from src.classification.classifier import OffensiveTextClassifier

train_comments = [
    "you are so stupid", "get lost idiot", "shut up moron",
    "have a nice day", "thanks so much", "you are very kind",
    "this is stupid", "you are an idiot", "great job today",
    "have a wonderful morning"
]
train_labels = [
    "offensive", "offensive", "offensive",
    "clean", "clean", "clean",
    "offensive", "offensive", "clean", "clean"
]

classifier = OffensiveTextClassifier()
classifier.train(train_comments, train_labels)

test_comments = ["you are stupid", "have a great day", "shut up"]
for comment in test_comments:
    label, probs = classifier.predict(comment)
    print(f"'{comment}' -> {label} | probabilities: {probs}")