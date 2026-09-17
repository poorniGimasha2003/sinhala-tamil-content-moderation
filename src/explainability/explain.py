def explain_prediction(classifier, comment: str, top_n=5):
    """
    Shows which words most strongly influenced the classifier's decision,
    by comparing each word's log-probability under 'offensive' vs 'clean'.
    """
    words = comment.lower().split()
    vectorizer = classifier.vectorizer
    model = classifier.model

    feature_names = vectorizer.get_feature_names_out()
    offensive_idx = list(model.classes_).index("offensive")
    clean_idx = list(model.classes_).index("clean")

    word_scores = []
    for word in words:
        if word in feature_names:
            feature_idx = list(feature_names).index(word)
            offensive_log_prob = model.feature_log_prob_[offensive_idx][feature_idx]
            clean_log_prob = model.feature_log_prob_[clean_idx][feature_idx]
            difference = offensive_log_prob - clean_log_prob
            word_scores.append((word, difference))

    word_scores.sort(key=lambda x: x[1], reverse=True)
    return word_scores[:top_n]