from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classifier(classifier, test_comments, test_labels):
    """
    Runs the classifier on the full test set and computes
    precision, recall, F1, and the confusion matrix.
    """
    predictions = [classifier.predict(comment)[0] for comment in test_comments]

    print("=== Classification Report ===")
    print(classification_report(test_labels, predictions))

    print("=== Confusion Matrix ===")
    labels = ["clean", "offensive"]
    cm = confusion_matrix(test_labels, predictions, labels=labels)
    print(f"{'':12}{'Predicted clean':18}{'Predicted offensive'}")
    for i, row_label in enumerate(labels):
        print(f"Actual {row_label:8}{cm[i][0]:<18}{cm[i][1]}")

    return predictions