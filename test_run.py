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