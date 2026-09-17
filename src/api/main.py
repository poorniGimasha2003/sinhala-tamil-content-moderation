from fastapi import FastAPI
from pydantic import BaseModel

from src.classification.data_loader import load_hate_speech_data, split_data
from src.classification.classifier import OffensiveTextClassifier
from src.explainability.explain import explain_prediction

app = FastAPI(title="Code-Mixed Content Moderation API")

# Train the classifier once, when the API starts up
comments, labels = load_hate_speech_data("data/raw/sentence-level-annotation.csv")
train_comments, test_comments, train_labels, test_labels = split_data(comments, labels)

classifier = OffensiveTextClassifier()
classifier.train(train_comments, train_labels)


class CommentRequest(BaseModel):
    text: str


@app.post("/moderate")
def moderate_comment(request: CommentRequest):
    label, probs = classifier.predict(request.text)
    top_words = explain_prediction(classifier, request.text)

    return {
        "comment": request.text,
        "prediction": label,
        "probabilities": probs,
        "top_contributing_words": [{"word": w, "score": round(s, 3)} for w, s in top_words]
    }


@app.get("/")
def root():
    return {"message": "Content moderation API is running. POST to /moderate to classify a comment."}