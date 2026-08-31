from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


class OffensiveTextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()

    def train(self, comments: list[str], labels: list[str]):
        """labels should be like ['offensive', 'clean', 'offensive', ...]"""
        X = self.vectorizer.fit_transform(comments)
        self.model.fit(X, labels)

    def predict(self, comment: str):
        X = self.vectorizer.transform([comment])
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        return prediction, dict(zip(self.model.classes_, probabilities))