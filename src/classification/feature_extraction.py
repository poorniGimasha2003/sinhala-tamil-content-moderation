from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_features(comments: list[str]):
    """
    Converts a list of text comments into TF-IDF numeric vectors.
    Returns the vectorizer (needed later to transform new comments)
    and the feature matrix.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(comments)
    return vectorizer, tfidf_matrix