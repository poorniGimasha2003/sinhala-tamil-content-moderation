from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np


MODEL_NAME = "xlm-roberta-base"

# Convert our string labels to numbers (transformers need numeric labels)
LABEL2ID = {"clean": 0, "offensive": 1}
ID2LABEL = {0: "clean", 1: "offensive"}


def prepare_dataset(comments, labels, tokenizer):
    """Tokenizes text and converts labels to numbers, in the format transformers expects."""
    encodings = tokenizer(comments, truncation=True, padding=True, max_length=64)
    numeric_labels = [LABEL2ID[label] for label in labels]
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": numeric_labels
    })


def train_transformer_classifier(train_comments, train_labels, test_comments, test_labels):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_dataset = prepare_dataset(train_comments, train_labels, tokenizer)
    test_dataset = prepare_dataset(test_comments, test_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir="./models/transformer_checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    return trainer, tokenizer

import torch


def predict_with_transformer(trainer, tokenizer, comments, batch_size=16):
    """Runs the fine-tuned transformer on a list of comments, returns predicted labels."""
    device = trainer.model.device
    trainer.model.eval()  # tell the model we're predicting, not training

    all_predictions = []

    with torch.no_grad():  # don't track gradients, saves memory
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            encodings = tokenizer(batch, truncation=True, padding=True, max_length=64, return_tensors="pt")
            encodings = {k: v.to(device) for k, v in encodings.items()}
            outputs = trainer.model(**encodings)
            batch_predictions = outputs.logits.argmax(dim=-1).tolist()
            all_predictions.extend(batch_predictions)

    return [ID2LABEL[p] for p in all_predictions]