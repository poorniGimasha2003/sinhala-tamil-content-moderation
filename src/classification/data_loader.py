import pandas as pd
from sklearn.model_selection import train_test_split


def load_hate_speech_data(csv_path: str):
    """
    Loads the Sinhala-English hate speech dataset and simplifies
    the 3-way label (Not offensive / Abusive / Hate-Inducing)
    into a 2-way label (clean / offensive).
    """
    df = pd.read_csv(csv_path)

    # Keep only the columns we need
    df = df[["Sentence", "Hate_speech"]]

    # Map 3 original labels -> 2 simplified labels
    label_map = {
        "Not offensive": "clean",
        "Abusive": "offensive",
        "Hate-Inducing": "offensive",
    }
    df["label"] = df["Hate_speech"].map(label_map)

    comments = df["Sentence"].tolist()
    labels = df["label"].tolist()

    return comments, labels

def split_data(comments, labels, test_size=0.2):
    """
    Splits comments/labels into train and test sets.
    test_size=0.2 means 20% held back for testing, 80% for training.
    """
    return train_test_split(
        comments, labels,
        test_size=test_size,
        random_state=42,       # ensures the same split every time you run it
        stratify=labels        # keeps the same clean/offensive ratio in both sets
    )