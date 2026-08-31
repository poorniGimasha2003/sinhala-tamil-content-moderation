from collections import defaultdict
import math

PAD = "##"  # boundary marker for characters, like <s></s> for words
VOCAB_SIZE = 26 * 26 * 26  # rough estimate of 3-letter combos, for Laplace smoothing


def get_trigrams(word):
    """Turn 'mama' into padded trigrams: ##m, #ma, mam, ama, ma#, a##"""
    padded = PAD + word.lower() + PAD
    return [padded[i:i + 3] for i in range(len(padded) - 2)]


def build_language_model(words):
    """Count trigram frequencies AND context (first 2 letters) frequencies."""
    trigram_counts = defaultdict(int)
    context_counts = defaultdict(int)
    for word in words:
        for tri in get_trigrams(word):
            trigram_counts[tri] += 1
            context_counts[tri[:2]] += 1
    return trigram_counts, context_counts


def word_log_probability(word, trigram_counts, context_counts):
    """Log-probability of a word under one language's model (Laplace smoothed)."""
    log_prob = 0.0
    for tri in get_trigrams(word):
        context = tri[:2]
        numerator = trigram_counts[tri] + 1
        denominator = context_counts[context] + VOCAB_SIZE
        log_prob += math.log(numerator / denominator)
    return log_prob


class LanguageIdentifier:
    def __init__(self, training_data: dict):
        self.models = {
            lang: build_language_model(words)
            for lang, words in training_data.items()
        }

    def predict(self, word: str):
        scores = {
            lang: word_log_probability(word, tri_counts, ctx_counts)
            for lang, (tri_counts, ctx_counts) in self.models.items()
        }
        best_lang = max(scores, key=scores.get)
        return best_lang, scores

    def predict_sentence(self, sentence: str):
        """Returns a list of (word, predicted_language) for each word in a sentence."""
        return [(word, self.predict(word)[0]) for word in sentence.split()]