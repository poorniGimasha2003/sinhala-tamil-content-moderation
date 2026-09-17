from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_sample_weight


class OffensiveTextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()

    def train(self, comments: list[str], labels: list[str], balance_classes=True):
        X = self.vectorizer.fit_transform(comments)

        if balance_classes:
            sample_weights = compute_sample_weight(class_weight="balanced", y=labels)
            self.model.fit(X, labels, sample_weight=sample_weights)
        else:
            self.model.fit(X, labels)

    def predict(self, comment: str):
        X = self.vectorizer.transform([comment])
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        return prediction, dict(zip(self.model.classes_, probabilities))